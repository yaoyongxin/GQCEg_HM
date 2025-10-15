# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 20:00:37 2024

@author: ichen
"""

import numpy as np
import pickle
from typing import (
    List,
    Optional,
    Tuple,
    Union
)

from qiskit.quantum_info  import Statevector
from qiskit.quantum_info import Operator
from qiskit.quantum_info import Pauli

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Pauli
#from qiskit.primitives import StatevectorEstimator
from qiskit.compiler import transpile
from qiskit import *
from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
)

from qiskit_aer import AerSimulator
#import matplotlib.pyplot as plt

def add_pauli_rotation_gate(
    qc: "QuantumCircuit",
    pauli_string: str,
    theta: float,
    decompose_rzz: bool = True
):
    """
    Appends a Pauli rotation gate to a QuantumCircuit.
    Convention for Pauli string ordering is opposite to the Qiskit convention.
    For example, in string "XYZ" Pauli "X" acts on the first qubit.

    Parameters
    ----------
    qc : "QuantumCircuit"
        Qiskit "QuantumCircuit" to which the Pauli rotation gate is appended.
    pauli_string : str
        Pauli string defining the rotation.
    theta : float
        Rotation angle.
    decompose_rzz : bool
        If decompose_rzz==True, all rzz gates are decompsed into cx-rz-cx.
        Otherwise, the final circuit contains rzz gates.

    Returns
    -------
    ansatz_adaptvqite : List[str]
        List of Pauli strings entering the ansatz.
    params_adaptvqite : List[float64]
        Parameters (angles) of the ansatz.
    """

    if qc.num_qubits != len(pauli_string):
        raise ValueError("Circuit and Pauli string are of different size")
    if all([pauli=='I' or pauli=='X' or pauli=='Y' or pauli=='Z'
            for pauli in pauli_string])==False:
        raise ValueError("Pauli string does not have a correct format")

    nontriv_pauli_list = [(i,pauli)
                        for i,pauli in enumerate(pauli_string) if pauli!='I']
    for (i,pauli) in nontriv_pauli_list:
        if pauli=='X':
            qc.h(i)
        if pauli=='Y':
            qc.sdg(i)
            qc.h(i)
    for list_ind in range(len(nontriv_pauli_list)-2):
        qc.cx(nontriv_pauli_list[list_ind][0],nontriv_pauli_list[list_ind+1][0])
    if decompose_rzz==True:
        qc.cx(
            nontriv_pauli_list[len(nontriv_pauli_list)-2][0],
            nontriv_pauli_list[len(nontriv_pauli_list)-1][0]
            )
        qc.rz(theta,nontriv_pauli_list[len(nontriv_pauli_list)-1][0])
        qc.cx(
            nontriv_pauli_list[len(nontriv_pauli_list)-2][0],
            nontriv_pauli_list[len(nontriv_pauli_list)-1][0]
            )
    if decompose_rzz==False:
        qc.rzz(
            theta,
            nontriv_pauli_list[len(nontriv_pauli_list)-2][0],
            nontriv_pauli_list[len(nontriv_pauli_list)-1][0]
            )
    for list_ind in reversed(range(len(nontriv_pauli_list)-2)):
        qc.cx(nontriv_pauli_list[list_ind][0],nontriv_pauli_list[list_ind+1][0])
    for (i,pauli) in nontriv_pauli_list:
        if pauli=='X':
            qc.h(i)
        if pauli=='Y':
            qc.h(i)
            qc.s(i)
    return qc


class QiskitCircuitGeneratorAVQITE:
    """
    Class for reading out the AVQITE ansatz and incar files, and constructing
    a Qiskit QuantumCircuit object representing the ansatz.

    Attributes
    ----------
    ansatz_filename : str
        File name of the ansatz file, including the path.
        For this implementation, "ansatz_inp.pkle" type of ansatz files is used.
    incar_filename : str
        File name of the incar file, including the path.
        Incar file is used to read out the reference state.
    """
    def __init__(
        self,
        ansatz_filename: str,
        incar_filename: str
    ):
        self._ansatz_filename = ansatz_filename
        self._incar_filename = incar_filename

        #Reads out the operator list and the parameter list of the ansatz.
        (self._ansatz_adaptvqite,
         self._params_ansatz) = self.read_adaptvqite_ansatz(ansatz_filename)

        #Reads out the number of qubits.
        self._num_qubits = len(self._ansatz_adaptvqite[0])

        #Reads out the incar file.
        with open(incar_filename) as fp:
            incar_content = fp.read()
        ref_st_r_pos = incar_content.find("ref_state")
        #Reads out the reference state from the incar file.
        self._ref_state = incar_content[
                            ref_st_r_pos+13:ref_st_r_pos+13+self._num_qubits
                            ]

        #Initializes a QuantumCircuit object.
        self._init_qc = QuantumCircuit(self._num_qubits)

        #If the reference state contains "1"s, adds corresponding bit-flips.
        if all([(el=='0') or (el=='1') for el in self._ref_state]):
            self._init_qc.x(
                [i for i,el in enumerate(self._ref_state) if el=='1']
            )
        else:
            raise ValueError(
                "Reference state is supposed to be a string of 0s and 1s"
            )


    def read_adaptvqite_ansatz(
        self,
        filename: str
    ):
        """
        Reads the ansatz from a file resulting from adaptvqite calculation.

        Parameters
        ----------
        filename : str
            Name of a file containing the results of adaptvqite calculation.
            Has to be given in .pkle format.

        Returns
        -------
        ansatz_adaptvqite : List[str]
            List of Pauli strings entering the ansatz.
        params_adaptvqite : List[float64]
            Parameters (angles) of the ansatz.
        """
        if filename[-5:] != '.pkle':
            raise ImportError("Ansatz file should be given in .pkle format")
        with open(filename, 'rb') as inp:
            data_inp = pickle.load(inp)
            ansatz_adaptvqite = data_inp[0]
            params_adaptvqite = data_inp[1]
        return ansatz_adaptvqite, params_adaptvqite


    def construct_ansatz_qc(
        self,
        decompose_rzz: bool
    ):
        """
        Constructs a "QuantumCircuit" representing the AVQITE ansatz.

        Parameters
        ----------
        decompose_rzz : bool
            If decompose_rzz==True, all rzz gates are decompsed into cx-rz-cx.
            Otherwise, the final circuit contains rzz gates.

        Returns
        -------
        qc : "QuantumCircuit"
            "QuantumCircuit" representing the AVQITE ansatz.
        """
        qc = self._init_qc.copy()

        #Adds (potentially multi-qubit) Pauli rotation gates to the circuit
        #from the ansatz using "add_pauli_rotation_gate" function.
        #Option "decompose_rzz=False" can be used to not decompose rzz gates
        #into cx-rz-cx
        for i, pauli_string in enumerate(self._ansatz_adaptvqite):
            theta = self._params_ansatz[i]
            qc = add_pauli_rotation_gate(
                qc,
                pauli_string,
                theta,
                decompose_rzz
            )
        return qc
    
model = QiskitCircuitGeneratorAVQITE(ansatz_filename='ansatz_inp_50.pkle', incar_filename='incar')
qc = model.construct_ansatz_qc(decompose_rzz=False)

basis_gates = ['x','z', 'rx', 'rzz', 'rz', 'cx']
qc_test = transpile(qc, basis_gates=basis_gates,optimization_level=3)

basis_gates = ['x','z', 'rx', 'rzz', 'rz']
qc_new = transpile(qc, basis_gates=basis_gates,optimization_level=3)

def add_noise(qc,p_init,p2,p1,p_meas):
    ### Secret Data
    #p_alpha=1e-4
    #p_d=1.5e-4
    #p_d2=7.5e-4
    #p_dep=8e-4
    #p_xx=1e-3
    #p_h=1.25e-3

    #p2=3e-3
    #p1=4e-4
    #p_init=0#4e-4

    Noise_Model={
        "init_error": pauli_error([('X',p_init), ('I', 1 - p_init)]),
        "meas_error": pauli_error([('X',p_meas), ('I', 1 - p_meas)]),
        "Two_q_err":  depolarizing_error(p2,2),
        "One_q_err": pauli_error([('X', p1/3), ('Y', p1/3), ('Z', p1/3), ('I', 1 - p1)]),
        }

    #Noise_Model={
    #    "init_error": pauli_error([('X',p_init), ('I', 1 - p_init)]),
        #"Two_q_err":  pauli_error([('X', p2/6), ('Y', p2/6), ('Z', p2/6), ('I', 1 - p2/2)]),
    #    "X_channel": pauli_error([('X',p_alpha), ('I', 1 - p_alpha)]),
    #    "Y_channel": pauli_error([('Y',p_alpha), ('I', 1 - p_alpha)]),
    #    "ZZ_chanel":pauli_error([('ZZ',p_xx), ('II', 1 - p_xx)]),
    #    "ZZ_chanel_h":pauli_error([('ZZ',p_h), ('II', 1 - p_h)]),
    #    "Z_channel_1": pauli_error([('Z',p_d), ('I', 1 - p_d)]),
    #    "Z_channel_2": pauli_error([('Z',p_d2), ('I', 1 - p_d2)]),
    #    "Z_channel_3": pauli_error([('Z',p_alpha), ('I', 1 - p_alpha)]),
    #    "One_dep": depolarizing_error(p_dep, 1),
    #    }

    In_bar=0
    qc_Size=qc.qregs[0]._size
    if qc.cregs:
        #print(qc.cregs[0]._size)
        c_Size=qc.cregs[0]._size
        qr=QuantumRegister(qc_Size, 'q')
        cr=ClassicalRegister(c_Size, 'c')
        qc_new = QuantumCircuit(qr,cr)
    else:
        qr=QuantumRegister(qc_Size, 'q')
        qc_new = QuantumCircuit(qr)

    for i in range(qc_Size):
        qc_new.append(Noise_Model["init_error"],[qr[i]])

    for CirIns in qc:
        op=CirIns.operation
        if op.num_qubits==1 and op.name!='measure':
            #qc_new.append(CirIns)
            #if In_bar%2 == 0:
            #    qc_new.append(Noise_Model["One_q_err"],[qr[CirIns.qubits[0]._index]])
            if op.name=='rz':
                qc_new.append(CirIns)
                qc_new.append(Noise_Model["One_q_err"],[qr[CirIns.qubits[0]._index]])
                #qc_new.append(Noise_Model["Z_channel_3"],[qr[CirIns.qubits[0]._index]])
                #qc_new.append(Noise_Model["One_dep"],[qr[CirIns.qubits[0]._index]])
                #qc_new.append(Noise_Model["Z_channel_1"],[qr[CirIns.qubits[0]._index]])

            elif op.name=='rx':
                qc_new.append(CirIns)
                qc_new.append(Noise_Model["One_q_err"],[qr[CirIns.qubits[0]._index]])
                #qc_new.append(Noise_Model["X_channel"],[qr[CirIns.qubits[0]._index]])
                #qc_new.append(Noise_Model["One_dep"],[qr[CirIns.qubits[0]._index]])
                #qc_new.append(Noise_Model["Z_channel_1"],[qr[CirIns.qubits[0]._index]])

            elif op.name=='ry':
                qc_new.append(CirIns)
                qc_new.append(Noise_Model["One_q_err"],[qr[CirIns.qubits[0]._index]])
                #qc_new.append(Noise_Model["Y_channel"],[qr[CirIns.qubits[0]._index]])
                #qc_new.append(Noise_Model["One_dep"],[qr[CirIns.qubits[0]._index]])
                #qc_new.append(Noise_Model["Z_channel_1"],[qr[CirIns.qubits[0]._index]])

            else:
                qc_new.append(CirIns)
                qc_new.append(Noise_Model["One_q_err"],[qr[CirIns.qubits[0]._index]])
                #qc_new.append(Noise_Model["One_dep"],[qr[CirIns.qubits[0]._index]])
                #qc_new.append(Noise_Model["Z_channel_1"],[qr[CirIns.qubits[0]._index]])

        elif op.num_qubits==1 and op.name=='measure':
            qc_new.append(Noise_Model["meas_error"],[qr[CirIns.qubits[0]._index]])
            qc_new.append(CirIns)
            #print(qc_new.cregs)
            #print(CirIns)
            #raise ValueError("Invalid Operation")

        elif op.num_qubits==2:
            if op.name=='rzz':
                qc_new.append(CirIns)
                if In_bar%2 == 0:
                    qc_new.append(Noise_Model["Two_q_err"],[qr[CirIns.qubits[0]._index],qr[CirIns.qubits[1]._index]])
                    #qc_new.append(Noise_Model["Two_q_err"],[qr[CirIns.qubits[0]._index]])
                    #qc_new.append(Noise_Model["Two_q_err"],[qr[CirIns.qubits[1]._index]])
            #    qc_new.append(Noise_Model["Two_q_err"],[qr[CirIns.qubits[0]._index],qr[CirIns.qubits[1]._index]])
                    #qc_new.append(Noise_Model["ZZ_chanel_h"],[qr[CirIns.qubits[0]._index],qr[CirIns.qubits[1]._index]])
                    #qc_new.append(Noise_Model["ZZ_chanel"],[qr[CirIns.qubits[0]._index],qr[CirIns.qubits[1]._index]])
                    #qc_new.append(Noise_Model["One_dep"],[qr[CirIns.qubits[0]._index]])
                    #qc_new.append(Noise_Model["One_dep"],[qr[CirIns.qubits[1]._index]])
                    #qc_new.append(Noise_Model["Z_channel_1"],[qr[CirIns.qubits[0]._index]])
                    #qc_new.append(Noise_Model["Z_channel_1"],[qr[CirIns.qubits[1]._index]])
            else:
                qc_new.append(CirIns)
                if In_bar%2 == 0:
                    qc_new.append(Noise_Model["Two_q_err"],[qr[CirIns.qubits[0]._index],qr[CirIns.qubits[1]._index]])
                    #qc_new.append(Noise_Model["Two_q_err"],[qr[CirIns.qubits[0]._index]])
                    #qc_new.append(Noise_Model["Two_q_err"],[qr[CirIns.qubits[1]._index]])
                    #qc_new.append(Noise_Model["One_dep"],[qr[CirIns.qubits[0]._index]])
                    #qc_new.append(Noise_Model["One_dep"],[qr[CirIns.qubits[1]._index]])
                    #qc_new.append(Noise_Model["Z_channel_1"],[qr[CirIns.qubits[0]._index]])
                    #qc_new.append(Noise_Model["Z_channel_1"],[qr[CirIns.qubits[1]._index]])
            #qc_new.append(Noise_Model["Two_q_err"],[qr[CirIns.qubits[0]._index]])
            #qc_new.append(Noise_Model["Two_q_err"],[qr[CirIns.qubits[1]._index]])
            #if In_bar%2 == 0:
            #    qc_new.append(Noise_Model["Two_q_err"],[qr[CirIns.qubits[0]._index],qr[CirIns.qubits[1]._index]])

        elif op.name=='barrier':
             qc_new.append(CirIns)
             In_bar+=1

        else:
            raise ValueError("Invalid Operation")
    return qc_new

unenc={}
for s in [0.1,0.2,0.5,1.0,2.0]: 
    p=3e-3*s
    p1=0.1*p*4/3
    p2=p
    p_init=0.1*p*4/3
    p_meas=p
    
    Noise_Model={
            "Spam_error": pauli_error([('X',p_init), ('I', 1 - p_init)]),
            "meas_error": pauli_error([('X',p_meas), ('I', 1 - p_meas)]),
            "Two_q_err":  depolarizing_error(p2,1),
            "One_q_err": pauli_error([('X', p1/3), ('Y', p1/3), ('Z', p1/3), ('I', 1 - p1)]),
            }
    
    P_str=['IIIIIIII','ZIIIIIII','YZYIIIII','XZXIIIII','YZZZYIII','XZZZXIII',
           'YZZZZZYI','XZZZZZXI','IIIIIIII','IIZIIIII','IIYZYIII','IIXZXIII',
            'IIYZZZYI','IIXZZZXI','IIIIIIII','IIIIZIII','IIIIYZYI','IIIIXZXI',
            'IIIIIIII','IIIIIIZI','IIIIIIII','IZIIIIII','ZIIIIIII','ZZIIIIII']
    
    count_dict={}
    for i,Pstr in enumerate(P_str):
        #qc_noise=qc_test.copy()
        if Pstr == 'IIIIIIII':
            continue
        qc_noise=add_noise(qc_test,p_init,p2,p1,p_meas)
        for j, string in enumerate(Pstr):
            if string=='Z':
                pass
            elif string=='Y':
                qc_noise.sdg(qc_noise.qregs[0][j])
                qc_noise.append(Noise_Model["One_q_err"],[qc_noise.qregs[0][j]])
                qc_noise.h(qc_noise.qregs[0][j])
                qc_noise.append(Noise_Model["One_q_err"],[qc_noise.qregs[0][j]])
    
            elif string=='X':
                qc_noise.h(qc_noise.qregs[0][j])
                qc_noise.append(Noise_Model["One_q_err"],[qc_noise.qregs[0][j]])
                
        for j in range(qc_noise.qregs[0]._size):
            qc_noise.append(Noise_Model["meas_error"],[qc_noise.qregs[0][j]])
        qc_noise.measure_all()
        nshots=1e5
        aer_simulator = AerSimulator(method='automatic')
        transpiled_circuit = transpile(qc_noise, aer_simulator)
        result = aer_simulator.run(transpiled_circuit, shots = nshots).result()
        counts = result.get_counts()
        count_dict[Pstr]=counts
    
    exp_val=[]
    for Pstr in P_str:
        if Pstr == 'IIIIIIII':
            continue
        pos_lst=[]
        for k, w in enumerate(Pstr):
            if w!='I':
                pos_lst.append(k)
        val=0 
        for key in count_dict[Pstr].keys():
            new_key = np.array(list(map(int,list(key[::-1]))))
            if np.sum(new_key[pos_lst])%2==1:
                val-=count_dict[Pstr][key]/nshots
            else:
                val+=count_dict[Pstr][key]/nshots
        exp_val.append(val)
    
    
    val1=0.5-exp_val[0]/2
    val2=exp_val[1]/4+exp_val[2]/4
    val3=exp_val[3]/4+exp_val[4]/4
    val4=exp_val[5]/4+exp_val[6]/4
    val5=0.5-exp_val[7]/2
    val6=exp_val[8]/4+exp_val[9]/4
    val7=exp_val[10]/4+exp_val[11]/4
    val8=0.5-exp_val[12]/2
    val9=exp_val[13]/4+exp_val[14]/4
    val10=0.5-exp_val[15]/2
    val11=0.25-exp_val[16]/4-exp_val[17]/4+exp_val[18]/4
    
    exp_vals=[val1,val2,val3,val4,val5,val6,val7,val8,val9,val10,val11]
    unenc[str(s)]=exp_vals
    
with open('Unenc_u_50_m.pickle', 'wb') as handle:
    pickle.dump(unenc, handle, protocol=pickle.HIGHEST_PROTOCOL)

def add_pauli_rotation_gate_2(
    qc: "QuantumCircuit",
    pauli_string: str,
    theta: float,
    decompose_rzz: bool = True
):
    """
    Appends a Pauli rotation gate to a QuantumCircuit.
    Convention for Pauli string ordering is opposite to the Qiskit convention.
    For example, in string "XYZ" Pauli "X" acts on the first qubit.

    Parameters
    ----------
    qc : "QuantumCircuit"
        Qiskit "QuantumCircuit" to which the Pauli rotation gate is appended.
    pauli_string : str
        Pauli string defining the rotation.
    theta : float
        Rotation angle.
    decompose_rzz : bool
        If decompose_rzz==True, all rzz gates are decompsed into cx-rz-cx.
        Otherwise, the final circuit contains rzz gates.

    Returns
    -------
    ansatz_adaptvqite : List[str]
        List of Pauli strings entering the ansatz.
    params_adaptvqite : List[float64]
        Parameters (angles) of the ansatz.
    """

    #if qc.num_qubits != len(pauli_string):
    #    raise ValueError("Circuit and Pauli string are of different size")
    if all([pauli=='I' or pauli=='X' or pauli=='Y' or pauli=='Z'
            for pauli in pauli_string])==False:
        raise ValueError("Pauli string does not have a correct format")

    nontriv_pauli_list = [(i,pauli)
                        for i,pauli in enumerate(pauli_string) if pauli!='I']
    for (i,pauli) in nontriv_pauli_list:
        if pauli=='X':
            qc.h(i)
        if pauli=='Y':
            qc.sdg(i)
            qc.h(i)
    for list_ind in range(len(nontriv_pauli_list)-2):
        qc.cx(nontriv_pauli_list[list_ind][0],nontriv_pauli_list[list_ind+1][0])
    if decompose_rzz==True:
        qc.cx(
            nontriv_pauli_list[len(nontriv_pauli_list)-2][0],
            nontriv_pauli_list[len(nontriv_pauli_list)-1][0]
            )
        qc.rz(theta,nontriv_pauli_list[len(nontriv_pauli_list)-1][0])
        qc.cx(
            nontriv_pauli_list[len(nontriv_pauli_list)-2][0],
            nontriv_pauli_list[len(nontriv_pauli_list)-1][0]
            )
    if decompose_rzz==False:
        qc.rzz(
            theta,
            nontriv_pauli_list[len(nontriv_pauli_list)-2][0],
            nontriv_pauli_list[len(nontriv_pauli_list)-1][0]
            )
    for list_ind in reversed(range(len(nontriv_pauli_list)-2)):
        qc.cx(nontriv_pauli_list[list_ind][0],nontriv_pauli_list[list_ind+1][0])
    for (i,pauli) in nontriv_pauli_list:
        if pauli=='X':
            qc.h(i)
        if pauli=='Y':
            qc.h(i)
            qc.s(i)
    #return qc


class QiskitCircuitGeneratorAVQITE1:
    """
    Class for reading out the AVQITE ansatz and incar files, and constructing
    a Qiskit QuantumCircuit object representing the ansatz.

    Attributes
    ----------
    ansatz_filename : str
        File name of the ansatz file, including the path.
        For this implementation, "ansatz_inp.pkle" type of ansatz files is used.
    incar_filename : str
        File name of the incar file, including the path.
        Incar file is used to read out the reference state.
    """
    def __init__(
        self,
        ansatz_filename: str,
        incar_filename: str
    ):
        self._ansatz_filename = ansatz_filename
        self._incar_filename = incar_filename

        #Reads out the operator list and the parameter list of the ansatz.
        (self._ansatz_adaptvqite,
         self._params_ansatz) = self.read_adaptvqite_ansatz(ansatz_filename)

        #Reads out the number of qubits.
        self._num_qubits = len(self._ansatz_adaptvqite[0])

        #Reads out the incar file.
        with open(incar_filename) as fp:
            incar_content = fp.read()
        ref_st_r_pos = incar_content.find("ref_state")
        #Reads out the reference state from the incar file.
        self._ref_state = incar_content[ref_st_r_pos+13:ref_st_r_pos+13+self._num_qubits]

        #Initializes a QuantumCircuit object.
        self._init_qc = QuantumCircuit(self._num_qubits+5)

        self._init_qc.h(0)
        for i in range(self._num_qubits+1):
            self._init_qc.cx(i,i+1)
        #If the reference state contains "1"s, adds corresponding bit-flips.
        if self._ref_state.count('1')%2 != 0:
            self._init_qc.x(0)
        if all([(el=='0') or (el=='1') for el in self._ref_state]):
            self._init_qc.x(
                [i+1 for i,el in enumerate(self._ref_state) if el=='1']
            )
        else:
            raise ValueError(
                "Reference state is supposed to be a string of 0s and 1s"
            )

    def read_adaptvqite_ansatz(
        self,
        filename: str
    ):
        """
        Reads the ansatz from a file resulting from adaptvqite calculation.

        Parameters
        ----------
        filename : str
            Name of a file containing the results of adaptvqite calculation.
            Has to be given in .pkle format.

        Returns
        -------
        ansatz_adaptvqite : List[str]
            List of Pauli strings entering the ansatz.
        params_adaptvqite : List[float64]
            Parameters (angles) of the ansatz.
        """
        if filename[-5:] != '.pkle':
            raise ImportError("Ansatz file should be given in .pkle format")
        with open(filename, 'rb') as inp:
            data_inp = pickle.load(inp)
            ansatz_adaptvqite = data_inp[0]
            params_adaptvqite = data_inp[1]
        return ansatz_adaptvqite, params_adaptvqite


    def construct_ansatz_qc(
        self,
        num_bar,
        decompose_rzz: bool
    ):
        """
        Constructs a "QuantumCircuit" representing the AVQITE ansatz.

        Parameters
        ----------
        decompose_rzz : bool
            If decompose_rzz==True, all rzz gates are decompsed into cx-rz-cx.
            Otherwise, the final circuit contains rzz gates.

        Returns
        -------
        qc : "QuantumCircuit"
            "QuantumCircuit" representing the AVQITE ansatz.
        """
        qc = self._init_qc.copy()

        #Adds (potentially multi-qubit) Pauli rotation gates to the circuit
        #from the ansatz using "add_pauli_rotation_gate" function.
        #Option "decompose_rzz=False" can be used to not decompose rzz gates
        #into cx-rz-cx
        new_str=[]
        for st in self._ansatz_adaptvqite:
            new_str.append('I'+st+'Z')
        len_bar=len(new_str)//num_bar
        bar_count=0
        for i, pauli_string in enumerate(new_str):
            if (i%len_bar==0) and (i!=0) and (bar_count != num_bar-1):
                qc.barrier()
                bar_count+=1
            theta = self._params_ansatz[i]
            add_pauli_rotation_gate_2(
                qc,
                pauli_string,
                theta,
                decompose_rzz
            )
        qc.barrier()
        return qc
    
def add_meas(qc,num,Af_meas_n):
    qc_Size=qc.qregs[0]._size
    qr=QuantumRegister(qc_Size, 'q')
    cr=ClassicalRegister(2*(num)+Af_meas_n*3, 'c')
    qc_new = QuantumCircuit(qr,cr)
    
    syn_count=0
    for CirIns in qc:
        op=CirIns.operation
        if op.name=='barrier':
            
            for i in range(qc_Size-3):
                qc_new.cx(qr[i],qr[-3])
            qc_new.h(qr[-2])
            for i in range(qc_Size-3):
                qc_new.cx(qr[-2],qr[i])
            qc_new.h(qr[-2])

            qc_new.measure(qr[-2],cr[2*syn_count])
            qc_new.measure(qr[-3],cr[2*syn_count+1])
            #qc_new.barrier()
            syn_count+=1
        else:
            qc_new.append(CirIns)
    return qc_new
            

def measure_op(qc,p_str,p_init,p2,p1):
    Noise_Model={
        "init_error": pauli_error([('X',p_init), ('I', 1 - p_init)]),
        "Two_q_err":  depolarizing_error(p2,2),
        "One_q_err": pauli_error([('X', p1/3), ('Y', p1/3), ('Z', p1/3), ('I', 1 - p1)]),
        }
    qc.h(qc.qregs[0][-1])
    qc.append(Noise_Model["One_q_err"],[qc.qregs[0][-1]])
    for i, w in enumerate(p_str):
        if w=='I':
            pass
        elif w=='X':
            qc.cx(qc.qregs[0][-1],qc.qregs[0][i])
            qc.append(Noise_Model["Two_q_err"],[qc.qregs[0][-1],qc.qregs[0][i]])
            #qc.append(Noise_Model["Two_q_err"],[qc.qregs[0][-1]])
            #qc.append(Noise_Model["Two_q_err"],[qc.qregs[0][i]])
        elif w=='Y':
            qc.cy(qc.qregs[0][-1],qc.qregs[0][i])
            qc.append(Noise_Model["Two_q_err"],[qc.qregs[0][-1],qc.qregs[0][i]])
            #qc.append(Noise_Model["Two_q_err"],[qc.qregs[0][-1]])
            #qc.append(Noise_Model["Two_q_err"],[qc.qregs[0][i]])
        elif w=='Z':
            qc.cz(qc.qregs[0][-1],qc.qregs[0][i])
            qc.append(Noise_Model["Two_q_err"],[qc.qregs[0][-1],qc.qregs[0][i]])
            #qc.append(Noise_Model["Two_q_err"],[qc.qregs[0][-1]])
            #qc.append(Noise_Model["Two_q_err"],[qc.qregs[0][i]])
        
    qc.h(qc.qregs[0][-1])
    qc.append(Noise_Model["One_q_err"],[qc.qregs[0][-1]])
    
def measure_stb(qc,p_init,p2,p1):
    Noise_Model={
        "init_error": pauli_error([('X',p_init), ('I', 1 - p_init)]),
        "Two_q_err":  depolarizing_error(p2,2),
        "One_q_err": pauli_error([('X', p1/3), ('Y', p1/3), ('Z', p1/3), ('I', 1 - p1)]),
        }
    size=qc.qregs[0]._size
    for i in range(size-3):
        qc.cx(qc.qregs[0][i],qc.qregs[0][-3])
        qc.append(Noise_Model["Two_q_err"],[qc.qregs[0][i],qc.qregs[0][-3]])
        #qc.append(Noise_Model["Two_q_err"],[qc.qregs[0][-1]])
        #qc.append(Noise_Model["Two_q_err"],[qc.qregs[0][i]])

    qc.h(qc.qregs[0][-2])
    qc.append(Noise_Model["One_q_err"],[qc.qregs[0][-2]])

    for i in range(size-3):
        qc.cx(qc.qregs[0][-2],qc.qregs[0][i])
        qc.append(Noise_Model["Two_q_err"],[qc.qregs[0][i],qc.qregs[0][-3]])
        #qc.append(Noise_Model["Two_q_err"],[qc.qregs[0][-1]])
        #qc.append(Noise_Model["Two_q_err"],[qc.qregs[0][i]])

    qc.h(qc.qregs[0][-2])
    qc.append(Noise_Model["One_q_err"],[qc.qregs[0][-2]])
    
N_mea=4
Af_meas_n=3

model = QiskitCircuitGeneratorAVQITE1(ansatz_filename='ansatz_inp_50.pkle', incar_filename='incar')
qc_ice = model.construct_ansatz_qc(N_mea,decompose_rzz=False)

qc_ice=add_meas(qc_ice,N_mea,Af_meas_n)
#basis_gates = ['x','z', 'h', 'rzz','cx']
#qc_ice = transpile(qc_ice, basis_gates=basis_gates, optimization_level=3)


P_str=["IIIIIIIIII","IZIIIIIIIZ","IYZYIIIIIZ","IXZXIIIIIZ","IYZZZYIIIZ","IXZZZXIIIZ",
      "IYZZZZZYIZ","IXZZZZZXIZ","IIIIIIIIII","IIIZIIIIIZ","IIIYZYIIIZ","IIIXZXIIIZ",
        "IIIYZZZYIZ","IIIXZZZXIZ","IIIIIIIIII","IIIIIZIIIZ","IIIIIYZYIZ","IIIIIXZXIZ",
        "IIIIIIIIII","IIIIIIIZIZ","IIIIIIIIII","IIZIIIIIIZ","IZIIIIIIIZ","IZZIIIIIII"]

p2=0
L_str=len(P_str[0])
enc_dict={}
pass_dict={}
for s in [0.1,0.2,0.5,1.0,2.0]:
    count_dict={}
    for i,Pstr in enumerate(P_str):
        #qc_ideal=qc_ice.copy()
        p=3e-3*s
        p1=0.1*p*4/3
        p2=p
        p_init=0.1*p*4/3
        p_meas=p
        
        Noise_Model={
        "init_error": pauli_error([('X',p_init), ('I', 1 - p_init)]),
        "meas_error": pauli_error([('X',p_meas), ('I', 1 - p_meas)]),
        "Two_q_err":  depolarizing_error(p2,2),
        "One_q_err": pauli_error([('X', p1/3), ('Y', p1/3), ('Z', p1/3), ('I', 1 - p1)]),
        }
        
        
        if Pstr == "IIIIIIIIII":
            continue
        qc_noise=add_noise(qc_ice,p_init,p2,p1,p_meas)
    
        for j in range(Af_meas_n):
            if j==Af_meas_n-1:
                measure_op(qc_noise,Pstr,p_init,p2,p1)
                qc_noise.append(Noise_Model["meas_error"],[qc_noise.qregs[0][-1]])
                qc_noise.measure(qc_noise.qregs[0][-1], qc_noise.cregs[0][-1*Af_meas_n])
            else:
                measure_op(qc_noise,Pstr,p_init,p2,p1)
                qc_noise.append(Noise_Model["meas_error"],[qc_noise.qregs[0][-1]])
                qc_noise.measure(qc_noise.qregs[0][-1], qc_noise.cregs[0][-1*(j+1)])
                measure_stb(qc_noise,p_init,p2,p1)
                qc_noise.append(Noise_Model["meas_error"],[qc_noise.qregs[0][-2]])
                qc_noise.measure(qc_noise.qregs[0][-2], qc_noise.cregs[0][2*N_mea+2*j])
                qc_noise.append(Noise_Model["meas_error"],[qc_noise.qregs[0][-3]])
                qc_noise.measure(qc_noise.qregs[0][-3], qc_noise.cregs[0][2*N_mea+2*j+1])
                
            
        nshots=1e5
        aer_simulator = AerSimulator(method='automatic')
        transpiled_circuit = transpile(qc_noise, aer_simulator)
        result = aer_simulator.run(transpiled_circuit, shots = nshots).result()
        counts = result.get_counts()
        count_dict[Pstr]=counts
    
    pass_rate=[]
    exp_val=[]
    for Pstr in P_str:
        if Pstr == "IIIIIIIIII":
            continue
        
        val=0
        num=0
        for key in count_dict[Pstr].keys():
            new_key = np.array(list(map(int,list(key[::-1]))))
            if np.sum(new_key[:-Af_meas_n])==0:
                num+=count_dict[Pstr][key]
                if np.sum(new_key[-Af_meas_n:])>(Af_meas_n-1)/2:
                    val-=count_dict[Pstr][key]
                else:
                    val+=count_dict[Pstr][key]
        exp_val.append(val/num)
        pass_rate.append(num/nshots)
        
    val1=0.5-exp_val[0]/2
    val2=exp_val[1]/4+exp_val[2]/4
    val3=exp_val[3]/4+exp_val[4]/4
    val4=exp_val[5]/4+exp_val[6]/4
    val5=0.5-exp_val[7]/2
    val6=exp_val[8]/4+exp_val[9]/4
    val7=exp_val[10]/4+exp_val[11]/4
    val8=0.5-exp_val[12]/2
    val9=exp_val[13]/4+exp_val[14]/4
    val10=0.5-exp_val[15]/2
    val11=0.25-exp_val[16]/4-exp_val[17]/4+exp_val[18]/4
    
    exp_vals=[val1,val2,val3,val4,val5,val6,val7,val8,val9,val10,val11]
    enc_dict[str(s)]=exp_vals
    pass_dict[str(s)]=pass_rate

with open('enc_val_4_3_u_50_m.pickle', 'wb') as handle:
    pickle.dump(enc_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('enc_suc_4_3_u_50_m.pickle', 'wb') as handle:
    pickle.dump(pass_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
val_lst=[0.5,-0.5,0.25,0.25,0.25,0.25,0.25,0.25,0.5,-0.5,0.25,0.25,0.25,0.25,0.5,-0.5,0.25,0.25,0.5,-0.5,0.25,-0.25,-0.25,0.25]

P_str_lst=['IIIIIIII','ZIIIIIII','YZYIIIII','XZXIIIII','YZZZYIII','XZZZXIII','YZZZZZYI','XZZZZZXI','IIIIIIII','IIZIIIII','IIYZYIII','IIXZXIII',
        'IIYZZZYI','IIXZZZXI',"IIIIIIII","IIIIZIII","IIIIYZYI","IIIIXZXI","IIIIIIII","IIIIIIZI","IIIIIIII","IZIIIIII","ZIIIIIII","ZZIIIIII"]
exp_vals=[]
for i,P_str in enumerate(P_str_lst):
    P = Pauli(P_str[::-1])
    op = Operator(P)
    if i%2==0:
        val=Statevector(qc_test).expectation_value(op)*val_lst[i]
    else:
        val+=Statevector(qc_test).expectation_value(op)*val_lst[i]
        exp_vals.append(val)

new_expval=exp_vals[:-2].copy()
new_expval.append(exp_vals[-1]+exp_vals[-2])
with open('EXT_expvals_u_50_m.npy', 'wb') as f:
    np.save(f, new_expval)