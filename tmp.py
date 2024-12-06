import json
import matplotlib.pyplot as plt

# tested both smooth and non-smooth, and they behave nicely
official_res = json.load(open('experiments/toy_example/results/logs_toyexampleNew.json'))
existing_res = json.load(open('../neuralProxies/experiments/toy_example/results/logs_toyexample_compareToOfficialCodeRepo.json'))

# now test with and without regTerm - works!
# official_res = json.load(open('experiments/toy_example/results/logs_toyexampleNewReg.json'))
# existing_res = json.load(open('../neuralProxies/experiments/toy_example/results/logs_toyexample_compareToOfficialCodeRepoReg.json'))

official_res = json.load(open('experiments/rosenbrock/results/logs_rosenbrock_gaussiansmoothsampler__bs4_kphi3_AT__seed0.json'))
existing_res = json.load(open('../neuralProxies/experiments/rosenbrock/results/logs_sigAsia_rosenbrock_gaussiansmoothsampler_NEURALproxy_bs4_kbeta3_AT_repoComparison_seed0.json'))

official_res = json.load(open('experiments/splines/results/logs_mnist_mlp_gaussiansampler__bs10_kphi10_AT__seed7.json'))
official_res = json.load(open('experiments/splines/results/logs_mnist_mlp_gaussiansmoothsampler__bs10_kphi10_AT_papersettings_seed7.json'))
existing_res = json.load(open('../neuralProxies/experiments/splines/results/logs_sigAsia_mnist_mlp_gaussiansampler_NEURALproxy_bs10_kbeta10_AT_sigg24_baselines5_seed7.json'))

# caustic:
official_res = json.load(open('experiments/caustic/results/logs_splinecaustic_gaussiansampler__bs10_kphi5_AT_test_seed0.json'))
official_res = json.load(open('experiments/caustic/results/logs_splinecaustic_gaussiansampler__bs10_kphi3_AT_paperSettings2_seed0.json'))
official_res = json.load(open('experiments/caustic/results/logs_splinecaustic_gaussiansampler__bs10_kphi5_AT_replRun_seed0.json'))
# official_res = json.load(open('experiments/caustic/results/logs_splinecaustic_gaussiansmoothsampler__bs10_kphi3_AT_paperSettings_wWarumup_seed0.json'))
existing_res = json.load(open('../neuralProxies/experiments/caustic/results/logs_sigAsia_splinecaustic_gaussiansampler_NEURALproxy_bs10_kbeta5_AT_sigg24_ourbaselines_idx5_seed0.json'))

# texture --> works
# official_res = json.load(open('experiments/texture/results/logs_texture_gaussiansmoothsampler__bs20_kphi3_AT_replicationTry_seed0.json'))   # replicationRun with "half" paper settings: 100k epochs, smooth=True, but lr 1e-4 instead of 1e-5
# official_res = json.load(open('experiments/texture/results/logs_texture_gaussiansmoothsampler__bs20_kphi3_AT_replicationTry_paperSettings_seed0.json'))   # true papersettings
# existing_res = json.load(open('../neuralProxies/experiments/texture/results/logs_sigAsia_texture_gaussiansampler_NEURALproxy_bs20_kbeta3_AT_run_our_baselines_bsSweep_seed0.json'))

# toy example, test if saleIndepLoss also beats L2 loss here -> yes, it does
# official_res = json.load(open('experiments/toy_example/results/logs_toyexampleNew_proxyL2.json'))
# existing_res = json.load(open('experiments/toy_example/results/logs_toyexampleNew_proxyScaleIndepLoss.json'))

# rosenbrock, test if scaleIndepLoss also beats L2 loss here -> no, it doesnt!
# official_res = json.load(open('experiments/rosenbrock/results/logs_rosenbrock_gaussiansmoothsampler__bs4_kphi3_AT_L2_seed0.json'))
# existing_res = json.load(open('experiments/rosenbrock/results/logs_rosenbrock_gaussiansmoothsampler__bs4_kphi3_AT_scaleIndepLoss_seed0.json'))



print(official_res.keys())

fig, ax = plt.subplots(1, 2)
ax[0].plot(official_res['img_errors'], label='New, Official Repo')
ax[0].plot(existing_res['img_errors'], label='Existing SIGGRAPH Repo', linestyle='dashed')

ax[1].plot(official_res['param_errors'], label='New, Official Repo')
ax[1].plot(existing_res['param_errors'], label='Existing SIGGRAPH Repo', linestyle='dashed')
[a.set_title(x) for a, x in zip(ax, ['Img Loss', 'Param Loss'])]
plt.legend()
plt.show()
