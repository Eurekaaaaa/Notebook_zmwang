import afterglowpy as grb
import emcee
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool

band_dict = {
    2.418e17: "1 keV",
    1.555e15: "UVW2",
    1.335e15: "UVM2",
    1.153e15: "UVW1",
    8.652e14: "U",
    8.443e14: "u",
    6.826e14: "B",
    6.389e14: "g",
    5.483e14: "V",
    4.862e14: "r",
    4.008e14: "i",
    3.356e14: "z",
    2.398e14: "J",
    1.851e14: "H",
    1.414e14: "Ks",
    1.000e10: "10 GHz",
    6.000e09: "6 GHz",
}


def interpret(band_dict, data):
    # Corrects for days
    if "days" in data.columns:
        data["t"] = data["days"] * 86400
    elif "seconds" in data.columns:
        data["t"] = data["seconds"]
    elif "t_delta" in data.columns:
        data["t"] = data["t_delta"]
    # Renames filter column if needed
    if "Filter" in data.columns:
        data["filter"] = data["Filter"]
    elif "Band" in data.columns:
        data["filter"] = data["Band"]
    elif "band" in data.columns:
        data["filter"] = data["band"]
    # Figures out the flux correction
    if "microJy" in data.columns:
        data["flux"] = data["microJy"]
        flux_correct = 1e-3
    elif "Jy" in data.columns:
        data["flux"] = data["Jy"]
        flux_correct = 1e3
    elif "mJy" in data.columns:
        data["flux"] = data["mJy"]
        flux_correct = 1
    elif "mag" in data.columns:
        data["flux"] = data["mag"]
        flux_correct = "mag"
    # Loops over dataframe and grabs errors and upper limits
    freq, new_flux, err = [], [], []
    for i in range(data.shape[0]):
        try:
            freq.append(
                list(band_dict.keys())[
                    list(band_dict.values()).index(data.iloc[i]["filter"])
                ]
            )
        except:
            freq.append("Unknown")
        if "err" in data.columns:
            flux = data.iloc[i]["flux"]
            error = float(data.iloc[i]["err"])
            if "<" in flux:
                new_flux.append("UL")
                err.append(0)
            elif ">" in flux:
                new_flux.append("UL")
                err.append(0)
            else:
                flux = float(flux)
                new_flux.append(flux)
                err.append(float(error))
        else:
            flux = data.iloc[i]["flux"]
            if "<" in flux:
                new_flux.append("UL")
                err.append(0)
            elif ">" in flux:
                new_flux.append("UL")
                err.append(0)
            elif "±" in flux:
                splt = flux.split("±")
                new_flux.append(float(splt[0]))
                err.append(float(splt[1]))
            elif "+-" in flux:
                splt = flux.split("+-")
                new_flux.append(float(splt[1]))
                err.append(float(splt[1]))
            elif "+" in flux or "-" in flux:
                splt = flux.split("+")
                new_flux.append(float(splt[0]))
                err_splt = splt[1].split("-")
                err.append(max([float(splt[0]), float(err_splt[1])]))
            else:
                new_flux.append(flux)
                err.append(0)

    for i in range(len(new_flux)):
        if new_flux[i] != "UL":
            if flux_correct != "mag":
                new_flux[i] = new_flux[i] * flux_correct
                err[i] = err[i] * flux_correct
            else:
                temp_flux = 1e3 * 3631 * 10 ** (float(new_flux[i]) / -2.5)
                max_flux = 1e3 * 3631 * 10 ** (float(new_flux[i] - err[i]) / -2.5)
                min_flux = 1e3 * 3631 * 10 ** (float(new_flux[i] + err[i]) / -2.5)
                new_flux[i] = 1e3 * 3631 * 10 ** (float(new_flux[i]) / -2.5)
                err[i] = max([max_flux - temp_flux, temp_flux - min_flux])
    data["frequency"] = freq
    data["flux"] = new_flux
    data["err"] = err

    data = data.loc[(data["flux"] != "UL") & (data["frequency"] != "Unknown")]

    data = data[["t", "frequency", "flux", "err"]].astype(np.float64)

    return data


def log_likelihood(theta, nu, x, y, yerr):
    E0, thetaObs, thetaCore, n0, epsilon_e, epsilon_B, p = theta
    Z = {
        "jetType": 0,  # Gaussian
        "specType": 0,  # Basic Synchrotron Emission Spectrum
        "thetaObs": thetaObs,  # Viewing angle in radians
        "E0": 10**E0,  # Isotropic-equivalent energy in erg
        "thetaCore": thetaCore,  # Half-opening angle in radians
        "thetaWing": 4 * thetaCore,  # "wing" truncation angle of the jet, in radians
        "n0": 10**n0,  # circumburst density in cm^-3
        "p": p,  # electron energy distribution index
        "b": 6.0,  # power law structure index
        "epsilon_e": 10**epsilon_e,  # epsilon_e
        "epsilon_B": 10**epsilon_B,  # epsilon_B
        "xi_N": 1.0,  # Fraction of electrons accelerated
        "d_L": 3.123611295013695e+27,  # Luminosity distance in cm
        "z": 0.2,
    }

    model = grb.fluxDensity(x, nu, **Z)
    sigma2 = yerr**2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log10(sigma2))


def log_prior(theta):
    E0, thetaObs, thetaCore, n0, epsilon_e, epsilon_B, p = theta
    if (
        0 <= thetaObs < np.pi * 0.5
        and 0.01 < thetaCore < np.pi * 0.5
        and 2 < p < 3
        and 0 < epsilon_e <= 1
        and 0 < epsilon_B <= 1
        and 0 < n0
    ):
        return 0.0
    return -np.inf


def log_probability(theta, nu, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, nu, x, y, yerr)


def plot_lc(data, theta):
    data = data.sort_values(by=["frequency"], ascending=False)
    filters = data["frequency"].unique()

    E0, thetaObs, thetaCore, n0, epsilon_e, epsilon_B, p = theta
    Z = {
        "jetType": 0,  # Gaussian
        "specType": 0,  # Basic Synchrotron Emission Spectrum
        "thetaObs": thetaObs,  # Viewing angle in radians
        "E0": E0,  # Isotropic-equivalent energy in erg
        "thetaCore": thetaCore,  # Half-opening angle in radians
        "thetaWing": 4 * thetaCore,  # "wing" truncation angle of the jet, in radians
        "n0": n0,  # circumburst density in cm^-3
        "p": p,  # electron energy distribution index
        "b": 6.0,  # power law structure index
        "epsilon_e": epsilon_e,  # epsilon_e
        "epsilon_B": epsilon_B,  # epsilon_B
        "xi_N": 1.0,  # Fraction of electrons accelerated
        "d_L": 3.123611295013695e+27,  # Luminosity distance in cm
        "z": 0.2,
    }

    times = np.geomspace(min(data["t"]), max(data["t"]), 100)

    fig, ax = plt.subplots()

    colours = plt.cm.viridis(np.linspace(0, 1, len(filters)))

    for i in range(len(filters)):
        to_plot = data.loc[data["frequency"] == filters[i]]
        nus = np.full(times.shape, filters[i])
        model = grb.fluxDensity(times, nus, **Z)
        ax.plot(times, model, color=colours[i], linewidth=0.5)
        ax.errorbar(
            to_plot["t"],
            to_plot["flux"],
            yerr=to_plot["err"],
            capsize=0,
            fmt=".",
            label=band_dict[filters[i]],
            color=colours[i],
        )
    ax.set(
        xlabel="Time since detection (s)",
        ylabel="Flux density (mJy)",
        xscale="log",
        yscale="log",
    )
    ax.legend(frameon=False)
    plt.show()


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path",
        type=str,
        help="path to the data file in the same format as files to be uploaded to GWAPA",
    )
    parser.add_argument(
        "-f", "--fit", action="store_true", help="fit the data using emcee"
    )
    parser.add_argument(
        "--params", action="store_true", help="print GWAPA parameter guess"
    )
    parser.add_argument(
        "--save_params",
        action="store_true",
        help="save GWAPA parameter guess to GWAPA_params.csv",
    )
    parser.add_argument(
        "--plot",
        type=str,
        help="plot the fitted light curve and save it to the given filepath",
        default=None,
    )
    parser.add_argument(
        "--niter",
        type=int,
        help="the number of MCMC iterations to perform",
        default=100
    )

    return parser


if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()

    Z = {
    "jetType": 0,  # Jet type
    "specType": 0,  # Basic Synchrotron Emission Spectrum
    "thetaObs": 0.2,  # Viewing angle in radians
    "E0": 1e+52,  # Isotropic-equivalent energy in erg
    "thetaCore": 0.25,  # Half-opening angle in radians
    "thetaWing": 1.0,  # "wing" truncation angle of the jet, in radians
    "n0": 0.01,  # circumburst density in cm^-3
    "p": 2.6,  # electron energy distribution index
    "b": 6.0,  # power law structure index
    "epsilon_e": 0.03162277660168379,  # epsilon_e
    "epsilon_B": 0.0031622776601683794,  # epsilon_B
    "xi_N": 1.0,  # Fraction of electrons accelerated
    "d_L": 3.123611295013695e+27,  # Luminosity distance in cm
    "z": 0.2,
    }

    if Z["jetType"] != 3:
        param_df = pd.DataFrame(
            [
                [
                    Z["E0"],
                    Z["thetaObs"],
                    Z["thetaCore"],
                    Z["n0"],
                    Z["epsilon_e"],
                    Z["epsilon_B"],
                    Z["p"],
                ]
            ],
            columns=[
                "E0",
                "thetaObs",
                "thetaCore",
                "n0",
                "epsilon_e",
                "epsilon_B",
                "p",
            ],
        )

    if args.params:
        print("Parameter guess from GWAPA:")
        print(param_df)

    if args.save_params:
        param_df.to_csv("GWAPA_params.csv", index=False)

    data = pd.read_csv(args.data_path)
    if data.shape[1] == 1:
        data = pd.read_csv(args.data_path, delim_whitespace=True)
    data = interpret(band_dict, data)

    if args.fit:
        if args.niter < 100:
            print("A minimum of 100 iterations must be used for the fitting.")
            exit()
        t = np.array(list(data["t"]))
        nu = np.array(list(data["frequency"]))
        fnu = np.array(list(data["flux"]))
        err = np.array(list(data["err"]))
        pos = [
            np.log10(Z["E0"]),
            Z["thetaObs"],
            Z["thetaCore"],
            np.log10(Z["n0"]),
            np.log10(Z["epsilon_e"]),
            np.log10(Z["epsilon_B"]),
            Z["p"],
        ] + 0.1 * np.random.randn(32, 7)

        nwalkers, ndim = pos.shape

        with Pool() as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_probability, args=(nu, t, fnu, err), pool=pool
            )

            sampler.run_mcmc(pos, args.niter, progress=True)

        flat_samples = sampler.get_chain(discard=10, thin=2, flat=True)

        theta = []

        params = ['log(E0)','thetaObs','thetaCore','log(n0)','log(epsilon_e)','log(epsilon_B)','p']

        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            if i in [0, 3, 4, 5]:
                theta.append(10 ** mcmc[1])
            else:
                theta.append(mcmc[1])
            q = np.diff(mcmc)
            print(f'{params[i]} = {mcmc[1]:.2f} +{q[1]:.2f} -{q[0]:.2f}')

    if args.plot:
        if args.fit:
            plot_lc(data, theta)
        else:
            plot_lc(
                data,
                (
                    Z["E0"],
                    Z["thetaObs"],
                    Z["thetaCore"],
                    Z["n0"],
                    Z["epsilon_e"],
                    Z["epsilon_B"],
                    Z["p"],
                ),
            )
