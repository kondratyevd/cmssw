import FWCore.ParameterSet.Config as cms

from HLTrigger.Configuration.common import *

def customizeOIseeding(process):
    """
    - adds doublet-like HB seeds (maxHitDoubletSeeds)
    - HL seeds from two trajectories (IP, MuS) are considered separate types
    - Number of seeds of each type can be determined individually for each L2 muon using a DNN
    """
    process.hltIterL3OISeedsFromL2Muons = cms.EDProducer( "TSGForOIDNN",
        MeasurementTrackerEvent = cms.InputTag("hltSiStripClusters"),
        SF1 = cms.double(3.0),
        SF2 = cms.double(4.0),
        SF3 = cms.double(5.0),
        SF4 = cms.double(7.0),
        SF5 = cms.double(10.0),
        SF6 = cms.double(2.0),
        adjustErrorsDynamicallyForHitless = cms.bool(True),
        debug = cms.untracked.bool(False),
        estimator = cms.string('hltESPChi2MeasurementEstimator100'),
        eta1 = cms.double(0.2),
        eta2 = cms.double(0.3),
        eta3 = cms.double(1.0),
        eta4 = cms.double(1.2),
        eta5 = cms.double(1.6),
        eta6 = cms.double(1.4),
        eta7 = cms.double(2.1),
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
        pT1 = cms.double(13.0),
        pT2 = cms.double(30.0),
        pT3 = cms.double(70.0),
        propagatorName = cms.string('PropagatorWithMaterialParabolicMf'),
        src = cms.InputTag("hltL2Muons","UpdatedAtVtx"),
        getStrategyFromDNN = cms.bool(True),  # will override max nSeeds of all types
        dnnMetadataPath = cms.string('RecoMuon/TrackerSeedGenerator/data/metadata.json')
    )

    return process
