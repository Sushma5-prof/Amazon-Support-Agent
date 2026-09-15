from sklearn.metrics import cohen_kappa_score


def calculate_agreement(
    annotator_a,
    annotator_b,
):
    if len(annotator_a) != len(annotator_b):
        raise ValueError(
            "Annotator lists must have the same length."
        )

    return cohen_kappa_score(
        annotator_a,
        annotator_b,
    )