import FWCore.ParameterSet.Config as cms

from HLTrigger.Configuration.common import *

def customizeOIseeding(process):
    """
    - adds doublet-like HB seeds (maxHitDoubletSeeds)
    - HL seeds from two trajectories (IP, MuS) are considered separate types
    - Number of seeds of each type and error SF for HL seeds  can be determined individually for each L2 muon using a DNN
    """
    process.hltIterL3OISeedsFromL2Muons = cms.EDProducer( "TSGForOIDNN",
        MeasurementTrackerEvent = cms.InputTag("hltSiStripClusters"),
        debug = cms.untracked.bool(False),
        estimator = cms.string('hltESPChi2MeasurementEstimator100'),
        fixedErrorRescaleFactorForHitless = cms.double(2.0),
        hitsToTry = cms.int32(1),
        layersToTry = cms.int32(2),
        maxEtaForTOB = cms.double(1.8),
        maxHitDoubletSeeds = cms.uint32(0),
        maxHitSeeds = cms.uint32(1),
        maxHitlessSeedsIP = cms.uint32(5),
        maxHitlessSeedsMuS = cms.uint32(0),
        maxSeeds = cms.uint32(20),
        minEtaForTEC = cms.double(0.7),
        propagatorName = cms.string('PropagatorWithMaterialParabolicMf'),
        src = cms.InputTag("hltL2Muons","UpdatedAtVtx"),
        getStrategyFromDNN = cms.bool(True),  # will override max nSeeds of all types and SF
        dnnMetadataPath = cms.string('RecoMuon/TrackerSeedGenerator/data/metadata.json')
    )

    return process
