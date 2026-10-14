"""Dataset lists used in the reproduction of Mantis v1 (arXiv 2502.15637v1).

Every list here is a deliberate, documented choice. Where the paper is silent,
the comment says so.
"""

# ---------------------------------------------------------------------------
# Archives (zero-shot evaluation, paper Figure 3 / Appendix Table 3: "159-D")
# ---------------------------------------------------------------------------

# Static copies of aeon 1.6's dataset lists, so the dataset set is pinned and the
# MOMENT environment (no aeon) can resolve names too.
UCR128_NAMES = [
    "ACSF1", "Adiac", "AllGestureWiimoteX", "AllGestureWiimoteY", "AllGestureWiimoteZ",
    "ArrowHead", "BME", "Beef", "BeetleFly", "BirdChicken", "CBF", "Car", "Chinatown",
    "ChlorineConcentration", "CinCECGTorso", "Coffee", "Computers", "CricketX",
    "CricketY", "CricketZ", "Crop", "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect", "DistalPhalanxTW",
    "DodgerLoopDay", "DodgerLoopGame", "DodgerLoopWeekend", "ECG200", "ECG5000",
    "ECGFiveDays", "EOGHorizontalSignal", "EOGVerticalSignal", "Earthquakes",
    "ElectricDevices", "EthanolLevel", "FaceAll", "FaceFour", "FacesUCR", "FiftyWords",
    "Fish", "FordA", "FordB", "FreezerRegularTrain", "FreezerSmallTrain", "Fungi",
    "GestureMidAirD1", "GestureMidAirD2", "GestureMidAirD3", "GesturePebbleZ1",
    "GesturePebbleZ2", "GunPoint", "GunPointAgeSpan", "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung", "Ham", "HandOutlines", "Haptics", "Herring",
    "HouseTwenty", "InlineSkate", "InsectEPGRegularTrain", "InsectEPGSmallTrain",
    "InsectWingbeatSound", "ItalyPowerDemand", "LargeKitchenAppliances", "Lightning2",
    "Lightning7", "Mallat", "Meat", "MedicalImages", "MelbournePedestrian",
    "MiddlePhalanxOutlineAgeGroup", "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW",
    "MixedShapesRegularTrain", "MixedShapesSmallTrain", "MoteStrain",
    "NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2", "OSULeaf", "OliveOil",
    "PLAID", "PhalangesOutlinesCorrect", "Phoneme", "PickupGestureWiimoteZ",
    "PigAirwayPressure", "PigArtPressure", "PigCVP", "Plane", "PowerCons",
    "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxTW", "RefrigerationDevices", "Rock", "ScreenType",
    "SemgHandGenderCh2", "SemgHandMovementCh2", "SemgHandSubjectCh2",
    "ShakeGestureWiimoteZ", "ShapeletSim", "ShapesAll", "SmallKitchenAppliances",
    "SmoothSubspace", "SonyAIBORobotSurface1", "SonyAIBORobotSurface2",
    "StarLightCurves", "Strawberry", "SwedishLeaf", "Symbols", "SyntheticControl",
    "ToeSegmentation1", "ToeSegmentation2", "Trace", "TwoLeadECG", "TwoPatterns", "UMD",
    "UWaveGestureLibraryAll", "UWaveGestureLibraryX", "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ", "Wafer", "Wine", "WordSynonyms", "Worms", "WormsTwoClass",
    "Yoga",
]

UEA30_NAMES = [
    "ArticularyWordRecognition", "AtrialFibrillation", "BasicMotions",
    "CharacterTrajectories", "Cricket", "DuckDuckGeese", "ERing", "EigenWorms",
    "Epilepsy", "EthanolConcentration", "FaceDetection", "FingerMovements",
    "HandMovementDirection", "Handwriting", "Heartbeat", "InsectWingbeat",
    "JapaneseVowels", "LSST", "Libras", "MotorImagery", "NATOPS", "PEMS-SF",
    "PenDigits", "PhonemeSpectra", "RacketSports", "SelfRegulationSCP1",
    "SelfRegulationSCP2", "SpokenArabicDigits", "StandWalkJump", "UWaveGestureLibrary",
]


def ucr128() -> list[str]:
    """All 128 UCR univariate datasets."""
    assert len(UCR128_NAMES) == 128
    return list(UCR128_NAMES)


# v1 text: three UEA datasets dropped (small test size or short sequences).
UEA_EXCLUDED = ["AtrialFibrillation", "StandWalkJump", "PenDigits"]


def uea27() -> list[str]:
    """UEA-30 minus the three datasets v1 excludes."""
    names = sorted(set(UEA30_NAMES) - set(UEA_EXCLUDED))
    assert len(names) == 27, f"expected 27 UEA datasets, got {len(names)}"
    return names


# v1: four further datasets complete the 159. Availability through aeon is not
# guaranteed; scripts skip and log any that fail to load.
OTHERS = ["Blink", "EMOPain", "MotionSenseHAR", "SharePriceIncrease"]

# v1 uses "InsectWingbeatSubset". How the subset was drawn is not described.
# Our choice: a stratified sample of this many train and test cases, seed 0.
INSECT_SUBSAMPLE = {"name": "InsectWingbeat", "n_train": 1000, "n_test": 1000, "seed": 0}


# ---------------------------------------------------------------------------
# Fine-tuning (paper Figure 5 / Appendix Table 4: "131-D")
# ---------------------------------------------------------------------------

# v1 excludes these 15 UCR datasets from fine-tuning (GPT4TS memory limits).
FT_EXCLUDED_UCR = [
    "EOGHorizontalSignal", "EOGVerticalSignal", "EthanolLevel", "HandOutlines",
    "InlineSkate", "MixedShapesRegularTrain", "PLAID", "PigAirwayPressure",
    "PigArtPressure", "PigCVP", "SemgHandGenderCh2", "SemgHandMovementCh2",
    "SemgHandSubjectCh2", "StarLightCurves", "UWaveGestureLibraryAll",
]

# Tier 1: 30 UCR datasets, all inside the paper's fine-tuning set (none are in
# FT_EXCLUDED_UCR). Chosen to spread across training-set size, series length,
# number of classes and data quirks. The approximate characteristics in the
# comments are from memory; scripts/00_dataset_stats.py prints the real ones.
TIER1 = [
    # very small training sets (<= 40 cases): where pretraining should matter most
    "BeetleFly",            # 20 train, len 512, 2 classes
    "Chinatown",            # 20 train, len 24 (very short)
    "TwoLeadECG",           # 23 train, len 82
    "DiatomSizeReduction",  # 16 train, 4 classes
    "Coffee",               # 28 train, spectrographs
    "Beef",                 # 30 train, 5 classes
    "CBF",                  # 30 train, synthetic
    "Rock",                 # 20 train, len 2844 (heavily downsampled to 512)
    "HouseTwenty",          # 40 train, len 2000 (downsampled)
    "CinCECGTorso",         # 40 train, len 1639 (downsampled)
    # small (41-200)
    "GunPoint",             # 50 train
    "Herring",              # 64 train
    "ItalyPowerDemand",     # 67 train, len 24 (very short)
    "Lightning7",           # 70 train, 7 classes
    "DodgerLoopDay",        # 78 train, MISSING VALUES
    "ECG200",               # 100 train
    "GesturePebbleZ1",      # 132 train, UNEQUAL LENGTH
    "Worms",                # 181 train, len 900
    # medium (201-1000)
    "Phoneme",              # 214 train, 39 classes, len 1024
    "Computers",            # 250 train, len 720
    "AllGestureWiimoteX",   # 300 train, UNEQUAL LENGTH, 10 classes
    "Earthquakes",          # 322 train, imbalanced
    "MedicalImages",        # 381 train, 10 classes
    "Adiac",                # 390 train, 37 classes
    "FiftyWords",           # 450 train, 50 classes
    "SwedishLeaf",          # 500 train, 15 classes
    "Strawberry",           # 613 train, spectrographs
    "Wafer",                # 1000 train, large test set
    # large (> 1000)
    "MelbournePedestrian",  # 1194 train, len 24, MISSING VALUES
    "FordA",                # 3601 train, len 500
]
assert len(TIER1) == 30 and not set(TIER1) & set(FT_EXCLUDED_UCR)


def tier2() -> list[str]:
    """Remaining UCR datasets of the paper's fine-tuning set (full fine-tuning, 1 seed)."""
    return sorted(set(ucr128()) - set(FT_EXCLUDED_UCR) - set(TIER1))


def resolve(selection: str) -> list[str]:
    """Turn a CLI selection into dataset names.

    Accepts: ucr, uea, others, all, tier1, tier2, or a comma-separated list of names.
    """
    sel = selection.strip().lower()
    if sel == "ucr":
        return ucr128()
    if sel == "uea":
        return uea27()
    if sel == "others":
        return list(OTHERS)
    if sel == "all":
        return ucr128() + uea27() + list(OTHERS)
    if sel == "tier1":
        return list(TIER1)
    if sel == "tier2":
        return tier2()
    return [s.strip() for s in selection.split(",") if s.strip()]
