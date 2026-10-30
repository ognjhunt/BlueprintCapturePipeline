"""Fixed identity dispatch; unchanged existing model generators retain their hashes."""
from . import deployment_embodiment_gofa_model as gofa


def for_identity(identity):
    from . import deployment_embodiment_kuka_model as kuka
    if identity == kuka.IDENTITY:
        return kuka
    from . import deployment_embodiment_yaskawa_model as yaskawa
    if identity in yaskawa.VARIANTS:
        return yaskawa
    if identity in gofa.PAYLOAD_KG:
        return gofa
    if identity in {name + "_robotiq_2f85_sim" for name in gofa.PAYLOAD_KG}:
        from . import deployment_embodiment_gofa_tool_model as gofa_tool
        return gofa_tool
    return None
