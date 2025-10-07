#!/usr/bin/env python
"""
A script to show how to create your own time domain source model.
A simple damped Gaussian signal is defined in the time domain, injected into
noise in two interferometers (LIGO Livingston and Hanford at design
sensitivity), and then recovered.
"""



import bilby
import numpy as np
from bilby.core.utils.random import seed
from bilby.core.utils import random

import sys 
sys.path.append('../../../../EOBnsmodes/external')
from copy import deepcopy
import eob_modes_bilby as eob

from tqdm import trange

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)


# define the time-domain model
def time_domain_damped_sinusoid(time, amplitude, damping_time, frequency, phase, t0):
    r"""
    This example only creates a linearly polarised signal with only plus
    polarisation.

    .. math::

        h_{\plus}(t) =
            \Theta(t - t_{0}) A
            e^{-(t - t_{0}) / \tau}
            \sin \left( 2 \pi f t + \phi \right)

    Parameters
    ----------
    time: array-like
        The times at which to evaluate the model. This is required for all
        time-domain models.
    amplitude: float
        The peak amplitude.
    damping_time: float
        The damping time of the exponential.
    frequency: float
        The frequency of the oscillations.
    phase: float
        The initial phase of the signal.
    t0: float
        The offset of the start of the signal from the start time.

    Returns
    -------
    dict:
        A dictionary containing "plus" and "cross" entries.

    """
    plus = np.zeros(len(time))
    tidx = time >= t0
    plus[tidx] = (
        amplitude
        * np.exp(-(time[tidx] - t0) / damping_time)
        * np.sin(2 * np.pi * frequency * (time[tidx] - t0) + phase)
    )
    cross = np.zeros(len(time))
    return {"plus": plus, "cross": cross}


# define parameters to inject.
injection_parameters = dict(
    luminosity_distance=20,
    mass_1=1.4,
    mass_2=1.4,
    chi_1=-0.4,
    chi_2=-0.2,
    r_M_init=24.,
    phi_init=0,
    iota=np.pi/2,
    lambda_tilde=450,
    delta_lambda_tilde=0,
    I1=eob.I_of_lambda(450),
    I2=eob.I_of_lambda(450),
    phase=0,
    ra=0,
    dec=0,
    psi=0,
    geocent_time=0.0,
    t0=0.0,
    fiducial=1
)

duration = 4
sampling_frequency = 4096
outdir = "outdir_relative_binning_moi"
label = "eob_time_domain_source_model_relative_binning_moi"

# call the waveform_generator to create our waveform model.
waveform = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model = eob.eob_waveform_model_fd_moi,
    parameter_conversion = lambda x : (x , []),
    start_time=injection_parameters["geocent_time"] - 0.5,
)
fiducial_parameters = injection_parameters.copy()

def conversion_func(params):
    converted_params = params.copy()
    lambda_1, lambda_2 = bilby.gw.conversion.lambda_tilde_delta_lambda_tilde_to_lambda_1_lambda_2(params["lambda_tilde"], params["delta_lambda_tilde"], params["mass_1"], params["mass_2"])
    converted_params["lambda_1"] = lambda_1
    converted_params["lambda_2"] = lambda_2
    return converted_params


# inject the signal into three interferometers
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 0.5,
)
ifos.inject_signal(
    waveform_generator=waveform, parameters=injection_parameters, raise_error=False
)

#  create the priors
prior = injection_parameters.copy()
prior.pop("fiducial")
prior["mass_1"] = 1.4
prior["mass_2"] = 1.4
prior["chi_1"] = -.4
prior["chi_2"] = -.2
prior["lambda_tilde"] = bilby.core.prior.Uniform( 200, 650, "lambda_tilde")
prior["delta_lambda_tilde"] = bilby.core.prior.Uniform(-300, 300, "delta_lambda_tilde")
prior["lambda_1"] = bilby.core.prior.Constraint(name="lambda_1", minimum=100, maximum=1000)
prior["lambda_2"] = bilby.core.prior.Constraint(name="lambda_2", minimum=100, maximum=1000)
prior["I1"] = bilby.core.prior.Uniform(10, 24, "I1")
prior["I2"] = bilby.core.prior.Uniform(10, 24, "I2")
prior["r_M_init"] = 24
prior["phi_init"] = 0.0
prior["luminosity_distance"] = 20
prior["iota"] = np.pi/2
prior["phase"] = 0
# define likelihood
prior = bilby.core.prior.PriorDict(prior, conversion_function=lambda params: conversion_func({**params, "mass_1": injection_parameters["mass_1"], "mass_2": injection_parameters["mass_2"]}))
likelihood = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(ifos, waveform, priors=prior, fiducial_parameters=fiducial_parameters,distance_marginalization=False)

# launch sampler
result = bilby.core.sampler.run_sampler(
    likelihood,
    prior,
    sampler="dynesty",
    npoints=500,
    walks=25,
    nact=2,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    result_class=bilby.gw.result.CBCResult,
)

alt_waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    #time_domain_source_model=eob.eob_waveform_model,
    frequency_domain_source_model=eob.eob_waveform_model_fd_no_fid_moi,
    parameter_conversion = lambda x : (x , []),
    start_time=injection_parameters["geocent_time"] - 0.5,
)
alt_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=alt_waveform_generator,
)
likelihood.distance_marginalization = False
weights = list()
for ii in trange(len(result.posterior)):
    parameters = dict(result.posterior.iloc[ii])
    likelihood.parameters.update(parameters)
    alt_likelihood.parameters.update(parameters)
    print("parameters are", parameters)
    print("alt_likelihood", alt_likelihood.log_likelihood_ratio())
    print("likelihood", likelihood.log_likelihood_ratio())
    weights.append(
        alt_likelihood.log_likelihood_ratio() - likelihood.log_likelihood_ratio()
    )
print("weights before exp are", weights)
weights = np.exp(weights)
print("weights are", weights)
print(
    f"Reweighting efficiency is {np.mean(weights)**2 / np.mean(weights**2) * 100:.2f}%"
)
print(f"Binned vs unbinned log Bayes factor {np.log(np.mean(weights)):.2f}")

# Generate result object with the posterior for the regular likelihood using
# rejection sampling
alt_result = deepcopy(result)
keep = weights > random.rng.uniform(0, max(weights), len(weights))
alt_result.posterior = result.posterior.iloc[keep]

# Make a comparison corner plot.
bilby.core.result.plot_multiple(
    [result, alt_result],
    labels=["Binned", "Reweighted"],
    filename=f"{outdir}/{label}_corner.png",
)



result.plot_corner()
