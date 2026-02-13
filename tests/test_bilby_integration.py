import os
import threading
import time
from signal import SIGINT

import bilby
import pytest


@pytest.fixture(autouse=True)
def bilby_test_mode(bilby_test_mode):
    pass


@pytest.fixture(scope="session")
def sampler():
    return "ultranest"


@pytest.fixture(scope="session")
def sampler_kwargs():
    return dict(nlive=100, temporary_directory=False)


@pytest.fixture
def outdir(tmp_path):
    return tmp_path / "outdir"


@pytest.fixture
def conversion_function():
    def _conversion_function(parameters, likelihood, prior):
        converted = parameters.copy()
        if "derived" not in converted:
            converted["derived"] = converted["x"] * converted["y"]
        return converted

    return _conversion_function


def run_sampler(
    likelihood, priors, outdir, conversion_function, sampler, npool=None, **kwargs
):
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler=sampler,
        outdir=str(outdir),
        save="hdf5",
        npool=npool,
        conversion_function=conversion_function,
        **kwargs,
    )
    return result


def test_run_sampler(
    bilby_gaussian_likelihood_and_priors,
    outdir,
    conversion_function,
    npool,
    sampler,
    sampler_kwargs,
):
    likelihood, priors = bilby_gaussian_likelihood_and_priors
    result = run_sampler(
        likelihood,
        priors,
        outdir,
        conversion_function,
        sampler,
        npool,
        **sampler_kwargs,
    )
    assert "derived" in result.posterior


def test_interrupt_sampler(
    bilby_gaussian_likelihood_and_priors,
    outdir,
    conversion_function,
    npool,
    sampler,
    sampler_kwargs,
):
    likelihood, priors = bilby_gaussian_likelihood_and_priors

    pid = os.getpid()

    def trigger_signal():
        # Let sampler setup begin, then interrupt this process.
        time.sleep(4)
        os.kill(pid, SIGINT)

    thread = threading.Thread(target=trigger_signal, daemon=True)
    thread.start()

    original_log_likelihood = likelihood.log_likelihood

    def slow_log_likelihood(parameters=None):
        time.sleep(0.01)
        return original_log_likelihood(parameters)

    likelihood.log_likelihood = slow_log_likelihood

    with pytest.raises((SystemExit, KeyboardInterrupt)) as exc:
        while True:
            run_sampler(
                likelihood,
                priors,
                outdir,
                conversion_function,
                sampler,
                npool=npool,
                exit_code=5,
                **sampler_kwargs,
            )

    if isinstance(exc.value, SystemExit):
        assert exc.value.code == 5
