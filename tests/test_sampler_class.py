import bilby
import pytest
from ultranest_bilby import Ultranest


@pytest.fixture()
def SamplerClass():
    return Ultranest


@pytest.fixture()
def create_sampler(SamplerClass, bilby_gaussian_likelihood_and_priors, tmp_path):
    likelihood, priors = bilby_gaussian_likelihood_and_priors

    def create_fn(**kwargs):
        return SamplerClass(
            likelihood,
            priors,
            outdir=tmp_path / "outdir",
            label="test",
            use_ratio=False,
            **kwargs,
        )

    return create_fn


@pytest.fixture
def sampler(create_sampler):
    return create_sampler()


def test_default_kwargs(sampler):
    expected = dict(
        resume="overwrite",
        show_status=True,
        num_live_points=None,
        wrapped_params=[0, 0],  # No parameters are wrapped
        derived_param_names=[],  # None is converted to []
        run_num=None,
        vectorized=False,
        num_test_samples=2,
        draw_multiple=True,
        num_bootstraps=30,
        update_interval_iter=None,
        update_interval_ncall=None,
        log_interval=None,
        dlogz=None,
        max_iters=None,
        update_interval_volume_fraction=0.2,
        viz_callback=None,
        dKL=0.5,
        frac_remain=0.01,
        Lepsilon=0.001,
        min_ess=400,
        max_ncalls=None,
        max_num_improvement_loops=-1,
        min_num_live_points=400,
        cluster_num_live_points=40,
        step_sampler=None,
    )

    sampler.kwargs["viz_callback"] = None  # Remove viz_callback for comparison
    assert sampler.kwargs == expected


def test_wrapped_parameters(
    SamplerClass, bilby_gaussian_likelihood_and_priors, tmp_path
):
    likelihood, priors = bilby_gaussian_likelihood_and_priors
    priors["x"] = bilby.core.prior.Uniform(0, 10, "x", boundary="periodic")
    priors["y"] = bilby.core.prior.Uniform(0, 10, "y", boundary="reflective")

    sampler = SamplerClass(
        likelihood,
        priors,
        outdir=tmp_path / "outdir",
        label="test",
        use_ratio=False,
    )

    assert sampler.kwargs["wrapped_params"] == [1, 0]


@pytest.mark.parametrize(
    "equiv",
    bilby.core.sampler.base_sampler.NestedSampler.npoints_equiv_kwargs,
)
def test_translate_kwargs(create_sampler, equiv):
    expected = dict(
        resume="overwrite",
        show_status=True,
        num_live_points=123,
        wrapped_params=[0, 0],  # No parameters are wrapped
        derived_param_names=[],
        run_num=None,
        vectorized=False,
        num_test_samples=2,
        draw_multiple=True,
        num_bootstraps=30,
        update_interval_iter=None,
        update_interval_ncall=None,
        log_interval=None,
        dlogz=None,
        max_iters=None,
        update_interval_volume_fraction=0.2,
        viz_callback=None,
        dKL=0.5,
        frac_remain=0.01,
        Lepsilon=0.001,
        min_ess=400,
        max_ncalls=None,
        max_num_improvement_loops=-1,
        min_num_live_points=400,
        cluster_num_live_points=40,
        step_sampler=None,
    )

    sampler = create_sampler(**{equiv: 123})
    sampler.kwargs["viz_callback"] = None  # Remove viz_callback for comparison

    assert sampler.kwargs == expected
