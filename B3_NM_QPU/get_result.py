from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
# job = service.job('cw5xsmepcbmg008zxsh0')
# job = service.job('cw5y7j6xa9wg008pb500')
# job = service.job('cw5yb746f0t0008vv9rg')    # L30
# job = service.job('cw6hd85bhxtg008wbz6g')   # L01
# job = service.job('cw6jntzbhxtg008wc460')     # L01, shots:
# job = service.job('cw6mty3ggr6g0087cyf0')     # L10, 1
# job = service.job('cw6n6qjjzdhg008e3fyg')     # L10, 2
# job = service.job('cw6ngn29ezk00080rccg')     # L10, 3
# job = service.job('cw6nsexjzdhg008e3jw0')     # L10, 2
# job = service.job('cw6nvd5bhxtg008wchr0')     # L10, 3
# job = service.job('cw6qj789ezk00080rn10')     # L10, 3
# job = service.job('cwdf0dh543p0008ehjmg')     # N07, 2
# job = service.job('cwdf8wk0r6b0008yf2t0')     # N07, 2
# job = service.job('cwdfa8s31we000878md0')     # N07, 3
# job = service.job('cwdfhtq543p0008ehps0')     # N07, 3
# job = service.job('cwdgjy39r49g008k5xe0')     # N07, 3, ibm_strasbourg
# job = service.job('cwdgtds9r49g008k5y70')     # N07, 3, ibm_fez
# job = service.job('cwefd0w543p00085vbc0')     # N07, estimator
# job = service.job('cweffzz40e000088bhjg')     # N07, estimator
# job = service.job('cweg4c131we00087exhg')     # N07, estimator, resiliance2
# job = service.job('cweg7kymptp00082aba0')     # N07, function, default
# job = service.job('cwegcvvmptp00082abv0')     # L30, function, default
# job = service.job('cwege7g543p00085vgx0')     # L30, function, default
# job = service.job('cwegy3040e000088bs7g')     # L30, function, level 2, ibm_brisbane
# job = service.job('cwenr709r49g0085p5hg')     # L10, function, level 2
# job = service.job('cweq2899r49g0085pf90')     # L30, function, level 3, ibm_brussels
# job = service.job('cwesbdd0r6b0008p6b1g')     # L30, estimation, level 1, ibm_brussels
# job = service.job('cwesdm631we00087gv10')     # L30, estimation, level 1, ibm_brussels
# job = service.job('cwetwas31we00087h2z0')     # L10, estimation, level 1
# job = service.job('cwevff5543p00085xrv0')     # L10, function, level 1
# job = service.job('cwevjzk31we00087h8p0')     # L10, estimator, level 2
# job = service.job('cwevmha0r6b0008p6sr0')     # L30, estimation, level 2
# job = service.job('cwevr00543p00085xv10')     # L10, function, level 2
# job = service.job('cwf443g31we00087k3kg')     # L10, function, level 3
# job = service.job('cwf4nr79r49g0085sav0')     # L10, function, level 2
# job = service.job('cwf5bhe9r49g0085sdeg')     # L10, function, level 1
# job = service.job('cwf446031we00087k3m0')     # L30, estimator, level 2
# job = service.job('cwf7jga0r6b0008p92yg')     # L10, function, level 2
# job = service.job('cwf7v2w0r6b0008p95h0')     # L10, function, level 2
# job = service.job('cwf8dwf31we00087kqdg')     # L10, function, level 3
# job = service.job('cwf9fv70r6b0008p9f5g')     # L10, function, level 3

# job = service.job('d0jb74wehmr0008gq33g')     # L30, level 1, estimator, ibm_torino
# job = service.job('d0jb7cnehmr0008gq360')     # L30, level 1, estimator, ibm_torino
# job = service.job('d0jb7q68jzxg008my0rg')     # L30, level 1, estimator, ibm_torino

# job = service.job('d0j7z8dehmr0008gp9q0')     # L30, level 2, estimator, ibm_torino
# job = service.job('d0j80gtvpqf00080rkhg')     # L30, level 2, estimator, ibm_torino
job = service.job('d0jb1ppfbx30008wgjvg')     # L30, level 2, estimator, ibm_torino
# job = service.job('d0jbvzf36cs0008rrkz0')     # L30, level 2, estimator, ibm_torino

# job = service.job('d0j81q6crrag008n4fd0')     # L10, function, level 2, estimator, ibm_torino
# job = service.job('d0j8220crrag008n4fgg')     # L10, function, level 2, estimator, ibm_torino

print(job.status())

job_result = job.result()

print(job_result)

for idx, eval in enumerate(job_result[0].data.evs):
    print(f"{idx}: {eval:.6f} std: {job_result[0].data.stds[idx]:.6f}")


print(job_result[0].data.evs_noise_factors)
