from omniisaacgymenvs.utils.usd_utils.create_instanceable_assets import convert_asset_instanceable

ASSET_USD_PATH = "/home/com-27x/OmniIsaacGymEnvs/omniisaacgymenvs/Robots_for_Omniverse/openUSD_assets/UnitreeRobotics/go1/go1.usd"
SOURCE_PRIM_PATH = "/"
SAVE_AS_PATH = "/home/com-27x/OmniIsaacGymEnvs/omniisaacgymenvs/Robots_for_Omniverse/openUSD_assets/UnitreeRobotics/go1/go1_instanceable.usd"

convert_asset_instanceable(
    asset_usd_path=ASSET_USD_PATH, 
    source_prim_path=SOURCE_PRIM_PATH, 
    save_as_path=SAVE_AS_PATH,
    #create_xforms=True
)