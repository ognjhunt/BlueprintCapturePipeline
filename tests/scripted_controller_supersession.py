"""Deliberate, reviewed revisions of pinned scripted-controller sources.

A predecessor snapshot proves that later body additions changed no existing
controller byte. A revision that intentionally changes one is listed here with
its reason; only that exact digest substitution is accepted.
"""
from __future__ import annotations

import json

SUPERSEDED = {
    # Accept the tool link Isaac Sim 6 authors for TRLC (tn__*_link67_*) from its
    # source name link6-7; scripted_robot_grasp_frames._SOURCE_TOOLS.
    "sha256:d0ae2f89fcb2218523dbbbb5f11beb812c8d2bd5e5dd1a892af97f8cee20347d":
        "sha256:9f3ed5b722aba7bbfd19952de6d1712683de88a0465bbe8b030326fcda9978e9",
    # The same revision re-pins each scripted catalog row's source_code_inventory digest.
    "sha256:b4184fd8f24332d0d5b93ccde09618a90b4af9a895bd0b217ff1a0ddce0cf036":
        "sha256:b4eded4f78c6a37bd906cf451c910f6052f4bdded696fee472edcbf3e0d8f48f",  # blueprint_scripted_agilex_piper_bimanual
    "sha256:108fbf410262bce11da27e6d37683b7c33d4635de6f53e992ee00a05c8bedc33":
        "sha256:b8572566e3c1b0e981c69395f6e55a4e3cd0cf6f4b072fcdb8aa612d559762a3",  # blueprint_scripted_almond_axol
    "sha256:0efdf1ca9c30b68a9bb9019f53bae4026dcbd6d0218cd707fbf73891b64dcd41":
        "sha256:4e31498371a52eda8112fa60ce04a6d0a6e15d5136bcc8d1f10bbdec050e918e",  # blueprint_scripted_franka_research_3_bimanual
    "sha256:50d3019e418c0af1709b0a5f53b88902e19053ab62632d10b2f9b9167baf741a":
        "sha256:df66a9385eec503d88ffe5b7fad92c5e655f7a9a020e63052e27381adadcc320",  # blueprint_scripted_i2rt_yam_bimanual
    "sha256:79017b53a06916d8289736455ad041e178d303da3aea565491b3b9cf4a7c565e":
        "sha256:ff586201d38b12dc3b1d2d17d247df82e5d47c6417a65ead8ef6f99bed0688eb",  # blueprint_scripted_kinova_gen3
    "sha256:460d2505d7279bcd7f512a6083733efd6b62a6d630b5279e920a6d78027f2449":
        "sha256:48d497a5ad773b19dc24b411db8e1b9cfffe87c1c291fd3e4b4cfa5f3e0de21a",  # blueprint_scripted_kinova_gen3_bimanual
    "sha256:a7d293ca17188ab12899947754ff985e2eeb918f84e6d958bd11db3f3832dc75":
        "sha256:42d9b1672673f992fb5b73da1613f2b5da420485bfc6af380a308fc114b405ff",  # blueprint_scripted_trlc_dk1_bimanual
    "sha256:2cb2ee39dffbb16962a6d800bfe202d5c906cbe7d8cf03514517c8b14acbe3b1":
        "sha256:d3dfe3dd4195138add33d5caf7963f1b009503d6fa576489d704c54428614de2",  # blueprint_scripted_ufactory_xarm7
    "sha256:7ed32f82ea3e9015ae90b91195644714aa5ed5c5039926dc2bc81818e5ae84c7":
        "sha256:310f6020b29845210d1cb5d42531d4921b62d3d98d7f738da4cff592a036a806",  # blueprint_scripted_ufactory_xarm7_bimanual
    "sha256:a98696b6352f3bcbb7897cfcfba18d3cea916bed263a26833b10002150e3502e":
        "sha256:090828566fbbcb9d69b0bff08e40bf132aa5600475aa1c3693ce94a691dac81d",  # blueprint_scripted_universal_robots_ur5e_bimanual
    "sha256:32c2f077f39ab7733cc29d01ddd9a2ba8f5fc0578b4e8f85862d86d764747cd8":
        "sha256:f30dbc32ebaa3642fcefd5d6c41d16a01f575f46627c0d5220ea4cc78ad0a4fc",  # blueprint_scripted_franka_research_3_robotiq_2f85_sim
    "sha256:a1d95c98390994f259f4686dedd1e0c06f8e3f205a094faffd6c6c0d60cc4698":
        "sha256:4b64b19e80bafed5cdc89fbfb2b3e74f14d3b311747f8c1470f582ec3fcba9fd",  # blueprint_scripted_universal_robots_ur10e_robotiq_2f85_sim
    "sha256:34080b13ff091337f243027a869730efa3aeca42e899e93f493755f530fb6934":
        "sha256:8a5f8af84a765f3d61d649dd7786ea461fda3bddc80366484e61700dc1310f5f",  # blueprint_scripted_universal_robots_ur5e_robotiq_2f85_sim
    "sha256:89b5a20bffc0415b2621560ebdedfe4926d9bf804cb6fad31e4ab8aaafc3d5a3":
        "sha256:7619487f271031c0ebce17797f832187e0a17bc9cfbc262f76cf7ba9b9ecb123",  # blueprint_scripted_fanuc_crx_10ia_robotiq_2f85_sim
    "sha256:17c3dd0bd688355d90e30a4833b187e62adc4f3d8e0770a65dff9469e1a01ff3":
        "sha256:d7385782438aafa70969215adbcbe9a6c03bb2c0cde4936947692815e2e54d2c",  # blueprint_scripted_fanuc_crx_10ia_l_robotiq_2f85_sim
    "sha256:714d9cc25d2999e80d0c6f10914bcf71ea6dea41e53f7af52c67960598945000":
        "sha256:3181909460c2ba89051ef14f8129e3406b7db51e2e419270d351740b3eb7c5a7",  # blueprint_scripted_fanuc_m710ic_45m_robotiq_2f85_sim
    "sha256:1486c8a0f68c4bc0d8421b5f2a55012b1f79b468e18827427afce6de641a0dee":
        "sha256:3df7558de340aa5ab6bd8c25d6ee86d9bb4dc1f2692d86e7f843c84ce6a4b632",  # blueprint_scripted_abb_gofa_crb15000_5_95_robotiq_2f85_sim
    "sha256:f790163c50f4faab20fe5865bcc3bee521d6be03871e56a28a4861eb06942661":
        "sha256:25c1cbc55ea51d06445951c327b5b77ba6c11280a030d46b4163f2742934efa5",  # blueprint_scripted_abb_gofa_crb15000_12_127_robotiq_2f85_sim
    "sha256:c4ac8e0afe89e717aac3c377cf490e318149ece4bd9ccaf6826fa57fdc8b4a50":
        "sha256:990885b9f557475752f928b203e817fa4a148a89c2a34247ed87ed3a22a86b15",  # blueprint_scripted_yaskawa_gp12_robotiq_2f85_sim
    "sha256:fd19946ebc011ddc24027d20ddcedad5346b4cf9ed3f2b3fb9f89c923204f487":
        "sha256:31c50683892e61a5578391267596632e1f9999f9d96d80f6c96e6fb7a463cdff",  # blueprint_scripted_yaskawa_gp20_robotiq_2f85_sim
    "sha256:54c7a768542c5a513ddb20ee458adc91b580931f2b1cb43ea5fb983c7a3a7d46":
        "sha256:af7f5e6224ecf08c77f5a8f87ef8931832e35304bb7ed00b4f2fc0461e98a35b",  # blueprint_scripted_kuka_lbr_iiwa_14_robotiq_2f85_sim
    "sha256:a12c014411600afb01da41553a6d6399bf3933e388a969c355d35ef405433d56":
        "sha256:f8047949cf5f693a1a04aa0937ef4d5197f45be76985c4fd417cd395ac1e40e9",  # blueprint_scripted_fanuc_m10id_12_robotiq_2f85_sim
    "sha256:1c27155db3c729d100946a98075d78d9f4028eba0e247afd200629f7cafafd22":
        "sha256:e4b852ffd0dd9d49a2ca8d9ae3138fb490edb0d9a1c2e80ed1b5ee51d32acf13",  # blueprint_scripted_yaskawa_hc20_robotiq_2f85_sim
    "sha256:dd9f015ceeb5de6d35a29684818ff6813fd91b12dfc1ae2352da5efe0139bab0":
        "sha256:91f03b0c66ad1e78adbc7f1890f80ec92f865c5209c6106bf6cc49b0cf3b7576",  # blueprint_scripted_agile_robots_diana_7_robotiq_2f85_sim
}


def current(snapshot):
    """The snapshot with each superseded digest replaced by its reviewed revision."""
    text = json.dumps(snapshot)
    for old, new in SUPERSEDED.items():
        text = text.replace(old, new)
    return json.loads(text)
