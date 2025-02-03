from costmodels import (
    MinimalisticCM,
    MinimalisticCMInput,
    MinimalisticCMOutput,
)


def test_minimalistic_cost_model():
    mcm = MinimalisticCM()

    cm_input = MinimalisticCMInput()

    cm_output = mcm.run(cm_input)

    assert isinstance(cm_output, MinimalisticCMOutput)

    cm_input.Area /= 2
    assert cm_input.Area < 65 * 10**6
    cm_output_small_area = mcm.run(cm_input)

    assert cm_output_small_area.LCoE > cm_output.LCoE
