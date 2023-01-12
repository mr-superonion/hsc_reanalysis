#!python
import os
import numpy as np
from builtins import zip
import astropy.io.fits as pyfits

import twopoint
from spec_tools import SpectrumInterp
from twopoint_cosmosis import theory_names
from cosmosis.gaussian_likelihood import GaussianLikelihood
from cosmosis.datablock import names

default_array = np.repeat(-1.0, 99)
psfsection = "psf_systematics_parameters"


def is_default(x):
    return len(x) == len(default_array) and (x == default_array).all()


def convert_nz_steradian(n):
    return n * (41253.0 * 60.0 * 60.0) / (4 * np.pi)


class TwoPointLikelihood(GaussianLikelihood):
    # This is a sub-class of the class GaussianLikelihood which can be found in
    # the file ${COSMOSIS_SRC_DIR}/cosmosis/gaussian_likelihood.py That
    # super-class implements the generic behaviour that all Gaussian
    # likelihoods follow - the basic form of the likelihoods, inverting
    # covariance matrices, saving results, etc.  This sub-clas does the parts
    # that are specific to this 2-pt likelihood - loading data from a file,
    # getting the specific theory prediction to which to compare it, etc.
    like_name = "2pt"

    def __init__(self, options):
        # We may decide to use an analytic gaussian covariance
        # in that case we won't load the covmat.
        super(TwoPointLikelihood, self).__init__(options)
        self.build_psfsys()

    def build_psfsys(self):
        print("Adding parameters for psf systematics (alpha and beta)")

        Dir = self.options.get_string("psf_dir")
        assert os.path.isdir(Dir), "Cannot find directory: %s!!" % Dir
        psf_ver = self.options.get_string("psf_ver")
        assert os.path.isfile(
            os.path.join(Dir, "star_gpgp_%s.fits" % psf_ver)
        ), "Cannot find star correlation files!!"

        self.pr = pyfits.getdata(os.path.join(Dir, "star_gpgp_%s.fits" % psf_ver))[
            "r_nom"
        ]
        # The seperations for PSF correlation functions are stored in units of arcmin
        self.pr = self.pr / 60.0 * np.pi / 180.0
        self.pp = pyfits.getdata(os.path.join(Dir, "star_gpgp_%s.fits" % psf_ver))[
            "xip"
        ]
        self.pq = pyfits.getdata(os.path.join(Dir, "star_gpgq_%s.fits" % psf_ver))[
            "xip"
        ]
        self.qq = pyfits.getdata(os.path.join(Dir, "star_gqgq_%s.fits" % psf_ver))[
            "xip"
        ]
        return

    def build_data(self):
        filename = self.options.get_string("data_file")
        self.save_plot_to = self.options.get_string("save_plot_to", default="")
        suffix = self.options.get_string("suffix", default="")
        if suffix:
            self.suffix = "_" + suffix
        else:
            self.suffix = suffix

        covmat_name = self.options.get_string("covmat_name", "COVMAT")
        # This is the main work - read data in from the file
        self.two_point_data = twopoint.TwoPointFile.from_fits(filename, covmat_name)

        # Potentially cut out lines. For some reason one version of
        # this file used zeros to mark masked values.
        if self.options.get_bool("cut_zeros", default=False):
            print("Removing 2-point values with value=0.0")
            self.two_point_data.mask_bad(0.0)

        if self.options.get_bool("cut_cross", default=False):
            print("Removing 2-point values from cross-bins")
            self.two_point_data.mask_cross()

        # All the names of two-points measurements that were found in the data
        # file
        all_names = [spectrum.name for spectrum in self.two_point_data.spectra]

        # We may not want to use all the likelihoods in the file.
        # We can set an option to only use some of them
        data_sets = self.options.get_string("data_sets", default="all")
        if data_sets != "all":
            data_sets = data_sets.split()
            self.two_point_data.choose_data_sets(data_sets)

        # The ones we actually used.
        self.used_names = [spectrum.name for spectrum in self.two_point_data.spectra]

        # Check for scale cuts. In general, this is a minimum and maximum angle
        # for each spectrum, for each redshift bin combination. Which is
        # clearly a massive pain... but what can you do?

        scale_cuts = {}
        for name in self.used_names:
            s = self.two_point_data.get_spectrum(name)
            for b1, b2 in s.bin_pairs:
                option_name = "angle_range_{}_{}_{}".format(name, b1, b2)
                if self.options.has_value(option_name):
                    r = self.options.get_double_array_1d(option_name)
                    scale_cuts[(name, b1, b2)] = r

        # Now check for completely cut bins
        # example:
        # cut_wtheta = 1,2  1,3  2,3
        bin_cuts = []
        for name in self.used_names:
            s = self.two_point_data.get_spectrum(name)
            option_name = "cut_{}".format(name)
            if self.options.has_value(option_name):
                cuts = self.options[option_name].split()
                cuts = [eval(cut) for cut in cuts]
                for b1, b2 in cuts:
                    bin_cuts.append((name, b1, b2))

        if scale_cuts or bin_cuts:
            self.two_point_data.mask_scales(scale_cuts, bin_cuts)
        else:
            print("No scale cuts mentioned in ini file.")

        # Info on which likelihoods we do and do not use
        print("Found these data sets in the file:")
        total_data_points = 0
        final_names = [spectrum.name for spectrum in self.two_point_data.spectra]
        for name in all_names:
            if name in final_names:
                data_points = len(self.two_point_data.get_spectrum(name))
            else:
                data_points = 0
            if name in self.used_names:
                print(
                    "    - {}  {} data points after cuts {}".format(
                        name, data_points, "  [using in likelihood]"
                    )
                )
                total_data_points += data_points
            else:
                print(
                    "    - {}  {} data points after cuts {}".format(
                        name, data_points, "  [not using in likelihood]"
                    )
                )
        print("Total data points used = {}".format(total_data_points))

        # Convert all units to radians.  The units in cosmosis are all
        # in radians, so this is the easiest way to compare them.
        for spectrum in self.two_point_data.spectra:
            if spectrum.is_real_space():
                spectrum.convert_angular_units("rad")

        # build up the data vector from all the separate vectors.
        # Just concatenation
        data_vector = np.concatenate(
            [spectrum.value for spectrum in self.two_point_data.spectra]
        )

        # Make sure
        if len(data_vector) == 0:
            raise ValueError(
                "No data was chosen to be used from 2-point data file {0}. \
                It was either not selectedin data_sets or cut out".format(
                    filename
                )
            )
        # The x data is not especially useful here, so return None.
        # We will access the self.two_point_data directly later to
        # determine ell/theta values
        return None, data_vector

    def build_covariance(self):

        C = np.array(self.two_point_data.covmat)
        r = self.options.get_int("covariance_realizations", default=-1)
        self.sellentin = self.options.get_bool("sellentin", default=False)

        if self.sellentin:
            if r < 0:
                print()
                print("ERROR: You asked for the Sellentin-Heavens corrections")
                print(
                    "by setting sellentin=T, but you did not set covariance_realizations"
                )
                print("If you want covariance_realizations=infinity you can use 0")
                print(
                    "(unlikely, but it's also possible you were super-perverse and set it negative?)"
                )
                print()
                raise ValueError(
                    "Please set covariance_realizations for 2pt like. See message above."
                )
            elif r == 0:
                print()
                print("NOTE: You asked for the Sellentin-Heavens corrections")
                print("but set covariance_realizations=0. I am assuming you want")
                print(
                    "the limit of an infinite number of realizations, so we will just go back"
                )
                print("to the original Gaussian model")
                print()
                self.sellentin = False
            else:
                # use proper correction
                self.covariance_realizations = r
                print()
                print(
                    "You set sellentin=T so I will apply the Sellentin-Heavens correction"
                )
                print("for a covariance matrix estimated from Monte-Carlo simulations")
                print("(you told us it was {} simulations in the ini file)".format(r))
                print(
                    "This analytic marginalization converts the Gaussian distribution"
                )
                print("to a multivariate student's t distribution instead.")
                print()
        elif r > 0:
            # Just regular increase in covariance size, no Sellentin change.
            p = C.shape[0]
            # This x is the inverse of the alpha used in the old code
            # because that applied to the weight matrix not the covariance
            x = (r - 1.0) / (r - p - 2.0)
            C = C * x
            print()
            print(
                "You set covariance_realizations={} in the 2pt likelihood parameter file".format(
                    r
                )
            )
            print(
                "So I will apply the Anderson-Hartlap correction to the covariance matrix"
            )
            print("The covariance matrix is nxn = {}x{}".format(p, p))
            print(
                "So the correction scales the covariance matrix by (r - 1) / (r - n - 2) = {}".format(
                    x
                )
            )
            print()
        return C

    def extract_theory_points(self, block):
        """Gets the theory prediction at the bin center. The code first updates
        the xip and xim to include PSF related systematics.
        """
        alpha = block.get_double(psfsection, "alpha_psf", default=0.0)
        beta = block.get_double(psfsection, "beta_psf", default=0.0)
        pp_spline = SpectrumInterp(self.pr, self.pp)
        qq_spline = SpectrumInterp(self.pr, self.qq)
        pq_spline = SpectrumInterp(self.pr, self.pq)

        n_a = block["shear_xi_plus", "nbin_a"]
        n_b = block["shear_xi_plus", "nbin_b"]
        theta = block["shear_xi_plus", "theta"]
        # only use data in the range from 3e-4 radians to 0.08 radians (1
        # arcmin~ 275 arcmin)
        mskT = np.greater(theta, 3e-4) & np.less(theta, 0.08)
        theta = theta[mskT]
        block["shear_xi_plus", "theta"] = theta
        block["shear_xi_minus", "theta"] = theta
        for i in range(n_a):
            for j in range(n_b):
                bname = "bin_{}_{}".format(i + 1, j + 1)
                if block.has_value("shear_xi_plus", bname):
                    block["shear_xi_plus", bname] = (
                        block["shear_xi_plus", bname][mskT]
                        + pp_spline(theta) * alpha**2.0
                        + pq_spline(theta) * 2.0 * alpha * beta
                        + qq_spline(theta) * beta**2.0
                    )
                    block["shear_xi_minus", bname] = block["shear_xi_minus", bname][
                        mskT
                    ]

        theory = []
        # We have a collection of data vectors, one for each spectrum
        # that we include. We concatenate them all into one long vector,
        # so we do the same for our theory data so that they match

        # We will also save angles and bin indices for plotting convenience,
        # although these are not actually used in the likelihood
        angle = []
        bin1 = []
        bin2 = []
        # dataset_name = []

        # Now we actually loop through our data sets
        for spectrum in self.two_point_data.spectra:
            (
                theory_vector,
                angle_vector,
                bin1_vector,
                bin2_vector,
            ) = self.extract_spectrum_prediction(block, spectrum)
            theory.append(theory_vector)
            angle.append(angle_vector)
            bin1.append(bin1_vector)
            bin2.append(bin2_vector)
            # dataset_name.append(np.repeat(spectrum.name, len(bin1_vector)))

        # We also collect the ell or theta values.
        # The gaussian likelihood code itself is not expecting these,
        # so we just save them here for convenience.
        angle = np.concatenate(angle)
        bin1 = np.concatenate(bin1)
        bin2 = np.concatenate(bin2)
        block[names.data_vector, self.like_name + "_angle"] = angle
        block[names.data_vector, self.like_name + "_bin1"] = bin1
        block[names.data_vector, self.like_name + "_bin2"] = bin2

        # the thing it does want is the theory vector, for comparison with
        # the data vector
        theory = np.concatenate(theory)
        return theory

    def extract_spectrum_prediction(self, block, spectrum):

        # We may need theory predictions for multiple different
        # types of spectra: e.g. shear-shear, pos-pos, shear-pos.
        # So first we find out from the spectrum where in the data
        # block we expect to find these - mapping spectrum types
        # to block names
        section, x_name, y_name = theory_names(spectrum)

        # To handle multiple different data sets we allow a suffix
        # to be applied to the section names, so that we can look up
        # e.g. "shear_cl_des" instead of just "shear_cl".
        section += self.suffix

        # We need the angle (ell or theta depending on the spectrum)
        # for the theory spline points - we will be interpolating
        # between these to get the data points
        angle_theory = block[section, x_name]

        # Now loop through the data points that we have.
        # For each one we have a pairs of bins and an angular value.
        # This assumes that we can take a single sample point from
        # each theory vector rather than integrating with a window function
        # over the theory to get the data prediction - this will need updating soon.
        bin_data = {}
        theory_vector = []

        # For convenience we will also return the bin and angle (ell or theta)
        # vectors for this bin too.
        angle_vector = []
        bin1_vector = []
        bin2_vector = []
        for (b1, b2, angle) in zip(spectrum.bin1, spectrum.bin2, spectrum.angle):
            # We are going to be making splines for each pair of values that we need.
            # We make splines of these and cache them so we don't re-make them for every
            # different theta/ell data point
            if (b1, b2) in bin_data:
                # either use the cached spline
                theory_spline = bin_data[(b1, b2)]
            else:
                # or make a new cache value load from the data block and make a
                # spline and save it
                if block.has_value(section, y_name.format(b1, b2)):
                    theory = block[section, y_name.format(b1, b2)]
                # It is okay to swap if the spectrum types are the same - symmetrical
                elif (
                    block.has_value(section, y_name.format(b2, b1))
                    and spectrum.type1 == spectrum.type2
                ):
                    theory = block[section, y_name.format(b2, b1)]
                else:
                    raise ValueError(
                        "Could not find theory prediction {} in section {}".format(
                            y_name.format(b1, b2), section
                        )
                    )
                theory_spline = SpectrumInterp(angle_theory, theory)
                bin_data[(b1, b2)] = theory_spline

            # use our spline - interpolate to this ell or theta value and add
            # to our list
            try:
                theory = theory_spline(angle)
            except ValueError:
                raise ValueError(
                    """Tried to get theory prediction for {} {}, \
                    but ell or theta value ({}) was out of range.
					"Maybe increase the range when computing/projecting or check \
                    units?""".format(
                        section, y_name.format(b1, b2), angle
                    )
                )
            theory_vector.append(theory)
            angle_vector.append(angle)
            bin1_vector.append(b1)
            bin2_vector.append(b2)

        # Return the whole collection as an array
        theory_vector = np.array(theory_vector)

        # For convenience we also save the angle vector (ell or theta) and bin
        # indices
        angle_vector = np.array(angle_vector)
        bin1_vector = np.array(bin1_vector, dtype=int)
        bin2_vector = np.array(bin2_vector, dtype=int)

        return theory_vector, angle_vector, bin1_vector, bin2_vector

    def do_likelihood(self, block):
        # Run the
        super(TwoPointLikelihood, self).do_likelihood(block)

        if self.sellentin:
            # The Sellentin-Heavens correction from arxiv 1511.05969
            # accounts for a finite number of Monte-Carlo realizations
            # being used to estimate the covariance matrix.

            # Note that this invalidates the saved simulation used for
            # the ABC sampler.  I can't think of a better way of doing this
            # than overwriting the whole things with NaNs - that will at
            # least make clear there is a problem somewhere and not
            # yield misleading results.
            block[names.data_vector, self.like_name + "_simulation"] = (
                np.nan * block[names.data_vector, self.like_name + "_simulation"]
            )

            # It changes the Likelihood from Gaussian to a multivariate
            # student's t distribution.  Here we will have to do a little
            # hack and overwrite the stuff that the original Gaussian
            # method did above
            N = self.covariance_realizations
            chi2 = block[names.data_vector, self.like_name + "_CHI2"]

            # We might be using a cosmologically varying
            # covariance matrix, though I'm not sure what that would mean.
            # There is a warning about this above.
            log_det = 0.0
            like = -0.5 * log_det - 0.5 * N * np.log(1 + chi2 / (N - 1.0))

            # overwrite the log-likelihood
            block[names.likelihoods, self.like_name + "_LIKE"] = like


setup, execute, cleanup = TwoPointLikelihood.build_module()
