from omniisaacgymenvs.utils.usd_utils.create_instanceable_assets import convert_asset_instanceable

ASSET_USD_PATH = "/home/com-27x/OmniIsaacGymEnvs/omniisaacgymenvs/asset/a1/omniverse_a1/a1_base.usd"
SOURCE_PRIM_PATH = "/a1_description"
SAVE_AS_PATH ="/home/com-27x/OmniIsaacGymEnvs/omniisaacgymenvs/asset/a1/omniverse_a1/a1_instanceable.usd"

convert_asset_instanceable(
    asset_usd_path=ASSET_USD_PATH, 
    source_prim_path=SOURCE_PRIM_PATH, 
    save_as_path=SAVE_AS_PATH,
    create_xforms=False
)



------------------------

from omniisaacgymenvs.utils.usd_utils.create_instanceable_assets import create_parent_xforms

ASSET_USD_PATH = "/home/com-27x/OmniIsaacGymEnvs/omniisaacgymenvs/asset/a1/omniverse_a1/a1.usd"
SOURCE_PRIM_PATH = "/a1_description"
SAVE_AS_PATH ="/home/com-27x/OmniIsaacGymEnvs/omniisaacgymenvs/asset/a1/omniverse_a1/a1_instanceable.usd"
create_parent_xforms(
    asset_usd_path=ASSET_USD_PATH, 
    source_prim_path=SOURCE_PRIM_PATH, 
    save_as_path=SAVE_AS_PATH
)