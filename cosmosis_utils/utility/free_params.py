from cosmosis.datablock import names, option_section as opt
import numpy as np

cosmo = names.cosmological_parameters
inv_neutrino_mass_fac = 0.010630291126160195


def setup(options):
    sample_choice = options.get_string(opt, "sample_choice", default="v2")
    return sample_choice


def execute(block, config):
    sample_choice = config
    # for unconstrained parameters
    mnu = block.get_double(cosmo, "mnu", default=0.06)
    if sample_choice == "v1":
        # sample omch2, ombh2, n_s
        omnuh2 = mnu * inv_neutrino_mass_fac
        # CDM, baryon and nu
        block[cosmo, "ommh2"] = block[cosmo, "omch2"] + block[cosmo, "ombh2"] + omnuh2
        block[cosmo, "h0"] = np.sqrt(block[cosmo, "ommh2"] / block[cosmo, "omega_m"])
        block[cosmo, "omega_b"] = block[cosmo, "ombh2"] / block[cosmo, "h0"] ** 2.0
    elif sample_choice == "v2":
        # sample h, ombh2, n_s
        block[cosmo, "omega_b"] = block[cosmo, "ombh2"] / block[cosmo, "h0"] ** 2.0
    elif sample_choice == "3x2pt":
        # sunao's 3x2pt prior
        # sample omch2, ombh2, omega_de
        block[cosmo, "A_s"] = np.exp(block[cosmo, "ln_as1e10"]) * 1e-10
        omk = block[cosmo, "omega_k"]
        ombh2 = block[cosmo, "ombh2"]
        omch2 = block[cosmo, "omch2"]
        omde = block[cosmo, "omega_de"]
        omnuh2 = mnu * inv_neutrino_mass_fac
        block[cosmo, "h0"] = (
            (ombh2 + omch2 + omnuh2) / (1.0 - omde - omk)
        ) ** 0.5
        block[cosmo, "omega_b"] = block[cosmo, "ombh2"] / block[cosmo, "h0"] ** 2.0
    elif sample_choice == "s16a":
        # block[cosmo,'mnu']= block[cosmo,'ommh2']/inv_neutrino_mass_fac
        block[cosmo, "A_s"] = 10.0 ** (block[cosmo, "log_as1e9"]) * 1e-9
    return 0


def cleanup(config):
    # nothing to do here!  We just include this
    # for completeness
    return 0
