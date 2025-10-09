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

import sys 
sys.path.append('../../../../EOBnsmodes/external')
import eob_modes_bilby as eob

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


def conversion_func(params):
    converted_params = params.copy()
    lambda_1, lambda_2 = bilby.gw.conversion.lambda_tilde_delta_lambda_tilde_to_lambda_1_lambda_2(params["lambda_tilde"], params["delta_lambda_tilde"], params["mass_1"], params["mass_2"])
    I1, I2 = eob.I_of_lambda(lambda_1) + params["delta_I1"], eob.I_of_lambda(lambda_2) + params["delta_I2"]
    converted_params["lambda_1"] = lambda_1
    converted_params["lambda_2"] = lambda_2
    converted_params["I1"] = I1
    converted_params["I2"] = I2
    return converted_params


# define parameters to inject.
injection_parameters = dict(
    luminosity_distance=10,
    mass_1=1.4,
    mass_2=1.4,
    chi_1=-0.4,
    chi_2=-0.2,
    r_M_init=24.,
    phi_init=0,
    iota=np.pi/2,
    lambda_tilde=450,
    delta_lambda_tilde=0,
    delta_I1 = 0.0,
    delta_I2 = 0.0,
    phase=0,
    ra=0,
    dec=0,
    psi=0,
    geocent_time=0.0,
    t0=0.0,
)

duration = 4
sampling_frequency = 4096
outdir = "outdir_moi_new"
label = "eob_time_domain_source_model"

# call the waveform_generator to create our waveform model.
waveform = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    time_domain_source_model=eob.eob_waveform_model_moi,
    parameter_conversion = lambda x : (x , []),
    start_time=injection_parameters["geocent_time"] + 2.0 - duration,
)

# inject the signal into three interferometers
ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] + 2.0 - duration,
)
ifos.inject_signal(
    waveform_generator=waveform, parameters=injection_parameters, raise_error=False
)

prior = injection_parameters.copy()
prior["mass_1"] = 1.4
prior["mass_2"] = 1.4
prior["chi_1"] = -.4
prior["chi_2"] = -.2
prior["lambda_tilde"] = bilby.core.prior.Uniform( 200, 650, "lambda_tilde")
prior["delta_lambda_tilde"] = bilby.core.prior.Uniform(-150, 150, "delta_lambda_tilde")
prior["lambda_1"] = bilby.core.prior.Constraint(name="lambda_1", minimum=100, maximum=1000)
prior["lambda_2"] = bilby.core.prior.Constraint(name="lambda_2", minimum=100, maximum=1000)
prior["delta_I1"] = bilby.core.prior.Uniform(-10, 10, "delta_I1")
prior["delta_I2"] = bilby.core.prior.Uniform(-10, 10, "delta_I2")
# Bound the I values to increase convergence 
prior["I1"] = bilby.core.prior.Constraint(name="I1", minimum=4, maximum=100)
prior["I2"] = bilby.core.prior.Constraint(name="I2", minimum=4, maximum=100)
prior["t0"] = bilby.core.prior.DeltaFunction(0.0, "t0")
prior["r_M_init"] = 24
prior["phi_init"] = 0.0
prior["luminosity_distance"] = 10
prior["iota"] = np.pi/2
prior["phase"] = 0

prior = bilby.core.prior.PriorDict(prior, conversion_function=lambda params: conversion_func({**params, "mass_1": injection_parameters["mass_1"], "mass_2": injection_parameters["mass_2"]}))
# define likelihood
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(ifos, waveform)
# launch sampler
# Call suggested by chatgpt to improve convergence
result = bilby.run_sampler(
    likelihood,
    prior,
    sampler="dynesty",
    npoints=500,
    walks=10,
    nact=2,
    dlogz=0.1,
    sample="rslice",
    use_dynesty_dynamic=True,
    npool=8,  # or however many cores you have
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    result_class=bilby.gw.result.CBCResult,
)
# Basic Sampler call
# result = bilby.core.sampler.run_sampler(
#     likelihood,
#     prior,
#     sampler="dynesty",
#     npoints=500,
#     walks=25,
#     nact=2,
#     injection_parameters=injection_parameters,
#     outdir=outdir,
#     label=label,
#     result_class=bilby.gw.result.CBCResult,
# )

result.plot_corner()
