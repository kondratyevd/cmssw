/**
  \class    TSGForOIDNN
  \brief    Create L3MuonTrajectorySeeds from L2 Muons in an outside-in manner
  \author   Dmitry Kondratyev, Arnab Purohit, Jan-Frederik Schulte (Purdue University, West Lafayette, USA)
 */

#include "RecoMuon/TrackerSeedGenerator/plugins/TSGForOIDNN.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include <memory>

TSGForOIDNN::TSGForOIDNN(const edm::ParameterSet& iConfig)
    : src_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      t_estimatorH_(esConsumes<Chi2MeasurementEstimatorBase, TrackingComponentsRecord>(edm::ESInputTag("", iConfig.getParameter<std::string>("estimator")))),
      t_magfieldH_(esConsumes<MagneticField, IdealMagneticFieldRecord>()),
      t_propagatorAlongH_(esConsumes<Propagator, TrackingComponentsRecord>(edm::ESInputTag("", iConfig.getParameter<std::string>("propagatorName")))),
      t_propagatorOppositeH_(esConsumes<Propagator, TrackingComponentsRecord>(edm::ESInputTag("", iConfig.getParameter<std::string>("propagatorName")))),
      t_tmpTkGeometryH_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
      t_geometryH_(esConsumes<GlobalTrackingGeometry, GlobalTrackingGeometryRecord>()),
      t_navSchool_(esConsumes<NavigationSchool, NavigationSchoolRecord>(edm::ESInputTag("", "SimpleNavigationSchool"))),
      t_SHPOpposite_(esConsumes<Propagator, TrackingComponentsRecord>(edm::ESInputTag("", "hltESPSteppingHelixPropagatorOpposite"))),
      maxSeeds_(iConfig.getParameter<uint32_t>("maxSeeds")),
      maxHitSeeds_(iConfig.getParameter<uint32_t>("maxHitSeeds")),
      maxHitlessSeeds_(iConfig.getParameter<uint32_t>("maxHitlessSeeds")),
      numOfLayersToTry_(iConfig.getParameter<int32_t>("layersToTry")),
      numOfHitsToTry_(iConfig.getParameter<int32_t>("hitsToTry")),
      fixedErrorRescalingForHitless_(iConfig.getParameter<double>("fixedErrorRescaleFactorForHitless")),
      adjustErrorsDynamicallyForHitless_(iConfig.getParameter<bool>("adjustErrorsDynamicallyForHitless")),
      minEtaForTEC_(iConfig.getParameter<double>("minEtaForTEC")),
      maxEtaForTOB_(iConfig.getParameter<double>("maxEtaForTOB")),
      updator_(new KFUpdator()),
      measurementTrackerTag_(
          consumes<MeasurementTrackerEvent>(iConfig.getParameter<edm::InputTag>("MeasurementTrackerEvent"))),
      pT1_(iConfig.getParameter<double>("pT1")),
      pT2_(iConfig.getParameter<double>("pT2")),
      pT3_(iConfig.getParameter<double>("pT3")),
      eta1_(iConfig.getParameter<double>("eta1")),
      eta2_(iConfig.getParameter<double>("eta2")),
      eta3_(iConfig.getParameter<double>("eta3")),
      eta4_(iConfig.getParameter<double>("eta4")),
      eta5_(iConfig.getParameter<double>("eta5")),
      eta6_(iConfig.getParameter<double>("eta6")),
      eta7_(iConfig.getParameter<double>("eta7")),
      SF1_(iConfig.getParameter<double>("SF1")),
      SF2_(iConfig.getParameter<double>("SF2")),
      SF3_(iConfig.getParameter<double>("SF3")),
      SF4_(iConfig.getParameter<double>("SF4")),
      SF5_(iConfig.getParameter<double>("SF5")),
      SF6_(iConfig.getParameter<double>("SF6")),
      theCategory_(std::string("Muon|RecoMuon|TSGForOIDNN")),
      maxHitlessSeedsIP_(iConfig.getParameter<uint32_t>("maxHitlessSeedsIP")),
      maxHitlessSeedsMuS_(iConfig.getParameter<uint32_t>("maxHitlessSeedsMuS")),
      maxHitDoubletSeeds_(iConfig.getParameter<uint32_t>("maxHitDoubletSeeds")),
      getStrategyFromDNN_(iConfig.getParameter<bool>("getStrategyFromDNN")),
      etaSplitForDnn_(iConfig.getParameter<double>("etaSplitForDnn")),
      dnnMetadataPath_(iConfig.getParameter<std::string>("dnnMetadataPath"))
      {
          if (getStrategyFromDNN_){
              edm::FileInPath dnnMetadataPath(dnnMetadataPath_);
              pt::read_json(dnnMetadataPath.fullPath(), metadata);
              tensorflow::setLogging("2");

              dnnModelPath_barrel_ = metadata.get<std::string>("barrel.dnnmodel_path");
              edm::FileInPath dnnPath_barrel(dnnModelPath_barrel_);
              graphDef_barrel_ = tensorflow::loadGraphDef(dnnPath_barrel.fullPath());
              tf_session_barrel_ = tensorflow::createSession(graphDef_barrel_);

              dnnModelPath_endcap_ = metadata.get<std::string>("endcap.dnnmodel_path");
              edm::FileInPath dnnPath_endcap(dnnModelPath_endcap_);
              graphDef_endcap_ = tensorflow::loadGraphDef(dnnPath_endcap.fullPath());
              tf_session_endcap_ = tensorflow::createSession(graphDef_endcap_);
          }
          produces<std::vector<TrajectorySeed> >();
      }

TSGForOIDNN::~TSGForOIDNN() {
    if (getStrategyFromDNN_){
        tensorflow::closeSession(tf_session_barrel_);
        tensorflow::closeSession(tf_session_endcap_);
        delete graphDef_barrel_;
        delete graphDef_endcap_;
    }
}

//
// Produce seeds
//
void TSGForOIDNN::produce(edm::StreamID sid, edm::Event& iEvent, edm::EventSetup const& iEventSetup) const {
    // Initialize variables
    unsigned int numSeedsMade = 0;
    unsigned int layerCount = 0;
    unsigned int hitlessSeedsMadeIP = 0;
    unsigned int hitlessSeedsMadeMuS = 0;
    unsigned int hitSeedsMade = 0;
    unsigned int hitDoubletSeedsMade = 0;

    // Surface used to make a TSOS at the PCA to the beamline
    Plane::PlanePointer dummyPlane = Plane::build(Plane::PositionType(), Plane::RotationType());

    // Read ESHandles
    edm::ESHandle<Chi2MeasurementEstimatorBase> estimatorH = iEventSetup.getHandle(t_estimatorH_);
    edm::ESHandle<MagneticField> magfieldH = iEventSetup.getHandle(t_magfieldH_);
    edm::ESHandle<Propagator> propagatorAlongH = iEventSetup.getHandle(t_propagatorAlongH_);
    edm::ESHandle<Propagator> propagatorOppositeH = iEventSetup.getHandle(t_propagatorOppositeH_);
    edm::ESHandle<TrackerGeometry> tmpTkGeometryH = iEventSetup.getHandle(t_tmpTkGeometryH_);
    edm::ESHandle<GlobalTrackingGeometry> geometryH = iEventSetup.getHandle(t_geometryH_);
    edm::ESHandle<NavigationSchool> navSchool = iEventSetup.getHandle(t_navSchool_);

    edm::Handle<MeasurementTrackerEvent> measurementTrackerH;
    iEvent.getByToken(measurementTrackerTag_, measurementTrackerH);

    // Read L2 track collection
    edm::Handle<reco::TrackCollection> l2TrackCol;
    iEvent.getByToken(src_, l2TrackCol);

    // The product
    std::unique_ptr<std::vector<TrajectorySeed> > result(new std::vector<TrajectorySeed>());

    // Get vector of Detector layers
    std::vector<BarrelDetLayer const*> const& tob = measurementTrackerH->geometricSearchTracker()->tobLayers();
    std::vector<ForwardDetLayer const*> const& tecPositive =
        tmpTkGeometryH->isThere(GeomDetEnumerators::P2OTEC)
        ? measurementTrackerH->geometricSearchTracker()->posTidLayers()
        : measurementTrackerH->geometricSearchTracker()->posTecLayers();
    std::vector<ForwardDetLayer const*> const& tecNegative =
        tmpTkGeometryH->isThere(GeomDetEnumerators::P2OTEC)
        ? measurementTrackerH->geometricSearchTracker()->negTidLayers()
        : measurementTrackerH->geometricSearchTracker()->negTecLayers();

    // Get suitable propagators
    std::unique_ptr<Propagator> propagatorAlong = SetPropagationDirection(*propagatorAlongH, alongMomentum);
    std::unique_ptr<Propagator> propagatorOpposite = SetPropagationDirection(*propagatorOppositeH, oppositeToMomentum);

    // Stepping Helix Propagator for propogation from muon system to tracker
    edm::ESHandle<Propagator> SHPOpposite = iEventSetup.getHandle(t_SHPOpposite_);

    // Loop over the L2's and make seeds for all of them
    LogTrace(theCategory_) << "TSGForOIDNN::produce: Number of L2's: " << l2TrackCol->size();
    for (unsigned int l2TrackColIndex(0); l2TrackColIndex != l2TrackCol->size(); ++l2TrackColIndex) {
        const reco::TrackRef l2(l2TrackCol, l2TrackColIndex);

        // Container of Seeds
        std::vector<TrajectorySeed> out;
        LogTrace("TSGForOIDNN") << "TSGForOIDNN::produce: L2 muon pT, eta, phi --> " << l2->pt() << " , " << l2->eta()
            << " , " << l2->phi() << std::endl;

        FreeTrajectoryState fts = trajectoryStateTransform::initialFreeState(*l2, magfieldH.product());

        dummyPlane->move(fts.position() - dummyPlane->position());
        TrajectoryStateOnSurface tsosAtIP = TrajectoryStateOnSurface(fts, *dummyPlane);
        LogTrace("TSGForOIDNN") << "TSGForOIDNN::produce: Created TSOSatIP: " << tsosAtIP << std::endl;

        // Get the TSOS on the innermost layer of the L2
        TrajectoryStateOnSurface tsosAtMuonSystem =
            trajectoryStateTransform::innerStateOnSurface(*l2, *geometryH, magfieldH.product());
        LogTrace("TSGForOIDNN") << "TSGForOIDNN::produce: Created TSOSatMuonSystem: " << tsosAtMuonSystem
            << std::endl;

        LogTrace("TSGForOIDNN")
            << "TSGForOIDNN::produce: Check the error of the L2 parameter and use hit seeds if big errors" << std::endl;

        StateOnTrackerBound fromInside(propagatorAlong.get());
        TrajectoryStateOnSurface outerTkStateInside = fromInside(fts);

        StateOnTrackerBound fromOutside(&*SHPOpposite);
        TrajectoryStateOnSurface outerTkStateOutside = fromOutside(tsosAtMuonSystem);

        // Check if the two positions (using updated and not-updated TSOS) agree withing certain extent.
        // If both TSOSs agree, use only the one at vertex, as it uses more information. If they do not agree, search for seeds based on both.
        double L2muonEta = l2->eta();
        double absL2muonEta = std::abs(L2muonEta);

        // make non-const copies of parameters
        // (we want to override them if DNN evaluation is enabled)
        unsigned int maxHitSeeds__ = maxHitSeeds_;
        unsigned int maxHitDoubletSeeds__ = maxHitDoubletSeeds_;
        unsigned int maxHitlessSeedsIP__ = maxHitlessSeedsIP_;
        unsigned int maxHitlessSeedsMuS__ = maxHitlessSeedsMuS_; 

        // update strategy parameters by evaluating DNN
        if (getStrategyFromDNN_){
            int nHBd(0), nHLIP(0), nHLMuS(0);
            bool dnnSuccess_ = false;

            // Put variables needed for DNN into an std::map
            std::map<std::string, float> feature_map_ = getFeatureMap(l2, tsosAtIP, outerTkStateOutside);

            if (std::abs(l2->eta())<etaSplitForDnn_){
                // barrel
                std::tie(nHBd, nHLIP, nHLMuS, dnnSuccess_) =  evaluateDnn(feature_map_, tf_session_barrel_, metadata.get_child("barrel") );
            } else {
                // endcap
                std::tie(nHBd, nHLIP, nHLMuS, dnnSuccess_) =  evaluateDnn(feature_map_, tf_session_endcap_, metadata.get_child("endcap") );
            }
            if (!dnnSuccess_) break;

            maxHitSeeds__ = 0;
            maxHitDoubletSeeds__ = nHBd;
            maxHitlessSeedsIP__ = nHLIP;
            maxHitlessSeedsMuS__ = nHLMuS;
        }

        numSeedsMade = 0;
        hitlessSeedsMadeIP = 0;
        hitlessSeedsMadeMuS = 0;
        hitSeedsMade = 0;
        hitDoubletSeedsMade = 0;

        // calculate scale factors
        double errorSFHitless =
            (adjustErrorsDynamicallyForHitless_ ? calculateSFFromL2(l2) : fixedErrorRescalingForHitless_);

        // BARREL
        if (absL2muonEta < maxEtaForTOB_) {
            layerCount = 0;
            for (auto it = tob.rbegin(); it != tob.rend(); ++it) {
                LogTrace("TSGForOIDNN") << "TSGForOIDNN::produce: looping in TOB layer " << layerCount << std::endl;

                if (hitlessSeedsMadeIP < maxHitlessSeedsIP__ && numSeedsMade < maxSeeds_)
                    makeSeedsWithoutHits(**it,
                                         tsosAtIP,
                                         *(propagatorAlong.get()),
                                         estimatorH,
                                         errorSFHitless,
                                         hitlessSeedsMadeIP,
                                         numSeedsMade,
                                         out);

                if (outerTkStateInside.isValid() && outerTkStateOutside.isValid() &&
                    hitlessSeedsMadeMuS < maxHitlessSeedsMuS__ && numSeedsMade < maxSeeds_)
                    makeSeedsWithoutHits(**it,
                                         outerTkStateOutside,
                                         *(propagatorOpposite.get()),
                                         estimatorH,
                                         errorSFHitless,
                                         hitlessSeedsMadeMuS,
                                         numSeedsMade,
                                         out);

                if (hitSeedsMade < maxHitSeeds__ && numSeedsMade < maxSeeds_)
                    makeSeedsFromHits(**it,
                                      tsosAtIP,
                                      *(propagatorAlong.get()),
                                      estimatorH,
                                      measurementTrackerH,
                                      hitSeedsMade,
                                      numSeedsMade,
                                      layerCount,
                                      out);

                if (hitDoubletSeedsMade < maxHitDoubletSeeds__ && numSeedsMade < maxSeeds_)
                    makeSeedsFromHitDoublets(**it,
                                             tsosAtIP,
                                             *(propagatorAlong.get()),
                                             estimatorH,
                                             measurementTrackerH,
                                             navSchool,
                                             hitDoubletSeedsMade,
                                             numSeedsMade,
                                             layerCount,
                                             out);

            }
            LogTrace("TSGForOIDNN") << "TSGForOIDNN:::produce: NumSeedsMade = " << numSeedsMade
                << " , layerCount = " << layerCount << std::endl;
        }

        // Reset number of seeds if in overlap region
        if (absL2muonEta > minEtaForTEC_ && absL2muonEta < maxEtaForTOB_) {
            numSeedsMade = 0;
            hitlessSeedsMadeIP = 0;
            hitlessSeedsMadeMuS = 0;
            hitSeedsMade = 0;
            hitDoubletSeedsMade = 0;
        }

        // ENDCAP+
        if (L2muonEta > minEtaForTEC_) {
            layerCount = 0;
            for (auto it = tecPositive.rbegin(); it != tecPositive.rend(); ++it) {
                LogTrace("TSGForOIDNN") << "TSGForOIDNN::produce: looping in TEC+ layer " << layerCount << std::endl;

                if (hitlessSeedsMadeIP < maxHitlessSeedsIP__ && numSeedsMade < maxSeeds_)
                    makeSeedsWithoutHits(**it,
                                         tsosAtIP,
                                         *(propagatorAlong.get()),
                                         estimatorH,
                                         errorSFHitless,
                                         hitlessSeedsMadeIP,
                                         numSeedsMade,
                                         out);

                if (outerTkStateInside.isValid() && outerTkStateOutside.isValid() &&
                    hitlessSeedsMadeMuS < maxHitlessSeedsMuS__ && numSeedsMade < maxSeeds_)
                    makeSeedsWithoutHits(**it,
                                         outerTkStateOutside,
                                         *(propagatorOpposite.get()),
                                         estimatorH,
                                         errorSFHitless,
                                         hitlessSeedsMadeMuS,
                                         numSeedsMade,
                                         out);

                if (hitSeedsMade < maxHitSeeds__ && numSeedsMade < maxSeeds_)
                    makeSeedsFromHits(**it,
                                      tsosAtIP,
                                      *(propagatorAlong.get()),
                                      estimatorH,
                                      measurementTrackerH,
                                      hitSeedsMade,
                                      numSeedsMade,
                                      layerCount,
                                      out);

                if (hitDoubletSeedsMade < maxHitDoubletSeeds__ && numSeedsMade < maxSeeds_)
                    makeSeedsFromHitDoublets(**it,
                                             tsosAtIP,
                                             *(propagatorAlong.get()),
                                             estimatorH,
                                             measurementTrackerH,
                                             navSchool,
                                             hitDoubletSeedsMade,
                                             numSeedsMade,
                                             layerCount,
                                             out);

            }
            LogTrace("TSGForOIDNN") << "TSGForOIDNN:::produce: NumSeedsMade = " << numSeedsMade
                << " , layerCount = " << layerCount << std::endl;
        }

        // ENDCAP-
        if (L2muonEta < -minEtaForTEC_) {
            layerCount = 0;
            for (auto it = tecNegative.rbegin(); it != tecNegative.rend(); ++it) {
                LogTrace("TSGForOIDNN") << "TSGForOIDNN::produce: looping in TEC- layer " << layerCount << std::endl;

                if (hitlessSeedsMadeIP < maxHitlessSeedsIP__ && numSeedsMade < maxSeeds_)
                    makeSeedsWithoutHits(**it,
                                         tsosAtIP,
                                         *(propagatorAlong.get()),
                                         estimatorH,
                                         errorSFHitless,
                                         hitlessSeedsMadeIP,
                                         numSeedsMade,
                                         out);

                if (outerTkStateInside.isValid() && outerTkStateOutside.isValid() &&
                    hitlessSeedsMadeMuS < maxHitlessSeedsMuS__ && numSeedsMade < maxSeeds_)
                    makeSeedsWithoutHits(**it,
                                         outerTkStateOutside,
                                         *(propagatorOpposite.get()),
                                         estimatorH,
                                         errorSFHitless,
                                         hitlessSeedsMadeMuS,
                                         numSeedsMade,
                                         out);

                if (hitSeedsMade < maxHitSeeds__ && numSeedsMade < maxSeeds_)
                    makeSeedsFromHits(**it,
                                      tsosAtIP,
                                      *(propagatorAlong.get()),
                                      estimatorH,
                                      measurementTrackerH,
                                      hitSeedsMade,
                                      numSeedsMade,
                                      layerCount,
                                      out);

                if (hitDoubletSeedsMade < maxHitDoubletSeeds__ && numSeedsMade < maxSeeds_)
                    makeSeedsFromHitDoublets(**it,
                                             tsosAtIP,
                                             *(propagatorAlong.get()),
                                             estimatorH,
                                             measurementTrackerH,
                                             navSchool,
                                             hitDoubletSeedsMade,
                                             numSeedsMade,
                                             layerCount,
                                             out);

            }
            LogTrace("TSGForOIDNN") << "TSGForOIDNN:::produce: NumSeedsMade = " << numSeedsMade
                << " , layerCount = " << layerCount << std::endl;
        }

        for (std::vector<TrajectorySeed>::iterator it = out.begin(); it != out.end(); ++it) {
            result->push_back(*it);
        }

    }  // L2Collection

    edm::LogInfo(theCategory_) << "TSGForOIDNN::produce: number of seeds made: " << result->size();

    iEvent.put(std::move(result));
}

//
// Create seeds without hits on a given layer (TOB or TEC)
//
void TSGForOIDNN::makeSeedsWithoutHits(const GeometricSearchDet& layer,
                                          const TrajectoryStateOnSurface& tsos,
                                          const Propagator& propagatorAlong,
                                          edm::ESHandle<Chi2MeasurementEstimatorBase>& estimator,
                                          double errorSF,
                                          unsigned int& hitlessSeedsMade,
                                          unsigned int& numSeedsMade,
                                          std::vector<TrajectorySeed>& out) const {
    // create hitless seeds
    LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsWithoutHits: Start hitless" << std::endl;
    std::vector<GeometricSearchDet::DetWithState> dets;
    layer.compatibleDetsV(tsos, propagatorAlong, *estimator, dets);
    if (!dets.empty()) {
        auto const& detOnLayer = dets.front().first;
        auto const& tsosOnLayer = dets.front().second;
        LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsWithoutHits: tsosOnLayer " << tsosOnLayer << std::endl;
        if (!tsosOnLayer.isValid()) {
            edm::LogInfo(theCategory_) << "ERROR!: Hitless TSOS is not valid!";
        } else {
            dets.front().second.rescaleError(errorSF);
            PTrajectoryStateOnDet const& ptsod =
                trajectoryStateTransform::persistentState(tsosOnLayer, detOnLayer->geographicalId().rawId());
            TrajectorySeed::RecHitContainer rHC;
            out.push_back(TrajectorySeed(ptsod, rHC, oppositeToMomentum));
            LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsWithoutHits: TSOS (Hitless) done " << std::endl;
            hitlessSeedsMade++;
            numSeedsMade++;
        }
    }
}

//
// Find hits on a given layer (TOB or TEC) and create seeds from updated TSOS with hit
//
void TSGForOIDNN::makeSeedsFromHits(const GeometricSearchDet& layer,
                                       const TrajectoryStateOnSurface& tsos,
                                       const Propagator& propagatorAlong,
                                       edm::ESHandle<Chi2MeasurementEstimatorBase>& estimator,
                                       edm::Handle<MeasurementTrackerEvent>& measurementTracker,
                                       unsigned int& hitSeedsMade,
                                       unsigned int& numSeedsMade,
                                       unsigned int& layerCount,
                                       std::vector<TrajectorySeed>& out) const {
    if (layerCount > numOfLayersToTry_)
        return;

    TrajectoryStateOnSurface onLayer(tsos);

    std::vector<GeometricSearchDet::DetWithState> dets;
    layer.compatibleDetsV(onLayer, propagatorAlong, *estimator, dets);

    // Find Measurements on each DetWithState
    LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsFromHits: Find measurements on each detWithState  "
        << dets.size() << std::endl;
    std::vector<TrajectoryMeasurement> meas;
    for (std::vector<GeometricSearchDet::DetWithState>::iterator it = dets.begin(); it != dets.end(); ++it) {
        MeasurementDetWithData det = measurementTracker->idToDet(it->first->geographicalId());
        if (det.isNull())
            continue;
        if (!it->second.isValid())
            continue;  // Skip if TSOS is not valid

        std::vector<TrajectoryMeasurement> mymeas =
            det.fastMeasurements(it->second, onLayer, propagatorAlong, *estimator);  // Second TSOS is not used
        for (std::vector<TrajectoryMeasurement>::const_iterator it2 = mymeas.begin(), ed2 = mymeas.end(); it2 != ed2;
             ++it2) {
            if (it2->recHit()->isValid())
                meas.push_back(*it2);  // Only save those which are valid
        }
    }

    // Update TSOS using TMs after sorting, then create Trajectory Seed and put into vector
    LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsFromHits: Update TSOS using TMs after sorting, then create "
        "Trajectory Seed, number of TM = "
        << meas.size() << std::endl;
    std::sort(meas.begin(), meas.end(), TrajMeasLessEstim());

    unsigned int found = 0;
    for (std::vector<TrajectoryMeasurement>::const_iterator it = meas.begin(); it != meas.end(); ++it) {
        TrajectoryStateOnSurface updatedTSOS = updator_->update(it->forwardPredictedState(), *it->recHit());
        LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsFromHits: TSOS for TM " << found << std::endl;
        if (not updatedTSOS.isValid())
            continue;

        edm::OwnVector<TrackingRecHit> seedHits;
        seedHits.push_back(*it->recHit()->hit());
        PTrajectoryStateOnDet const& pstate =
            trajectoryStateTransform::persistentState(updatedTSOS, it->recHit()->geographicalId().rawId());
        LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsFromHits: Number of seedHits: " << seedHits.size()
            << std::endl;
        TrajectorySeed seed(pstate, std::move(seedHits), oppositeToMomentum);
        out.push_back(seed);
        found++;
        numSeedsMade++;
        hitSeedsMade++;
        if (found == numOfHitsToTry_)
            break;
        if (hitSeedsMade > maxHitSeeds_)
            return;
    }

    if (found)
        layerCount++;
}



void TSGForOIDNN::makeSeedsFromHitDoublets(const GeometricSearchDet& layer,
                                              const TrajectoryStateOnSurface& tsos,
                                              const Propagator& propagatorAlong,
                                              edm::ESHandle<Chi2MeasurementEstimatorBase>& estimator,
                                              edm::Handle<MeasurementTrackerEvent>& measurementTracker,
                                              edm::ESHandle<NavigationSchool> navSchool,
                                              unsigned int& hitDoubletSeedsMade,
                                              unsigned int& numSeedsMade,
                                              unsigned int& layerCount,
                                              std::vector<TrajectorySeed>& out) const {

    // This method is similar to makeSeedsFromHits, but the seed is created
    // only when in addition to a hit on a given layer, there are more compatible hits
    // on next layers (going from outside inwards), compatible with updated TSOS.
    // If that's the case, multiple compatible hits are used to create a single seed.

    // Configured to only check the immideately adjacent layer and add one more hit
    int max_addtnl_layers = 1; // max number of additional layers to scan
    int max_meas = 1; // number of measurements to consider on each additional layer

    // // // First, regular procedure to find a compatible hit - like in makeSeedsFromHits // // //

    TrajectoryStateOnSurface onLayer(tsos);

    // Find dets compatible with original TSOS
    std::vector< GeometricSearchDet::DetWithState > dets;
    layer.compatibleDetsV(onLayer, propagatorAlong, *estimator, dets);

    LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsFromHitDoublets: Find measurements on each detWithState  " << dets.size() << std::endl;
    std::vector<TrajectoryMeasurement> meas;
    
    // Loop over dets
    for (std::vector<GeometricSearchDet::DetWithState>::iterator idet=dets.begin(); idet!=dets.end(); ++idet) {
        MeasurementDetWithData det = measurementTracker->idToDet(idet->first->geographicalId());

        if (det.isNull()) continue;    // skip if det does not exist
        if (!idet->second.isValid()) continue;    // skip if TSOS is invalid

        // Find measurements on this det
        std::vector <TrajectoryMeasurement> mymeas = det.fastMeasurements(idet->second, onLayer, propagatorAlong, *estimator);
    
        // Save valid measurements 
        for (std::vector<TrajectoryMeasurement>::const_iterator imea = mymeas.begin(), ed2 = mymeas.end(); imea != ed2; ++imea) {
            if (imea->recHit()->isValid()) meas.push_back(*imea);
        } // end loop over meas
    } // end loop over dets

    LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsFromHitDoublets: Update TSOS using TMs after sorting, then create Trajectory Seed, number of TM = " << meas.size() << std::endl;

    // sort valid measurements found on the first layer
    std::sort(meas.begin(), meas.end(), TrajMeasLessEstim());

    unsigned int found = 0;
    int hit_num = 0;

    // Loop over all valid measurements compatible with original TSOS
    for (std::vector<TrajectoryMeasurement>::const_iterator mea=meas.begin(); mea!=meas.end(); ++mea) {
        hit_num++;

        // Update TSOS with measurement on first considered layer
        TrajectoryStateOnSurface updatedTSOS = updator_->update(mea->forwardPredictedState(), *mea->recHit());

        LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsFromHitDoublets: TSOS for TM " << found << std::endl;
        if (not updatedTSOS.isValid()) continue;    // Skip if updated TSOS is invalid

        edm::OwnVector<TrackingRecHit> seedHits;

        // Save hit on first layer
        seedHits.push_back(*mea->recHit()->hit());
        const DetLayer* detLayer = dynamic_cast<const DetLayer*>(&layer);


        // // // Now for this measurement we will loop over additional layers and try to update the TSOS again // // //

        // find layers compatible with updated TSOS
        auto const& compLayers = (*navSchool).nextLayers(*detLayer, *updatedTSOS.freeState(), alongMomentum);

        int addtnl_layers_scanned=0;
        int found_compatible_on_next_layer = 0;
        int det_id = 0;

        // Copy updated TSOS - we will update it again with a measurement from the next layer, if we find it
        TrajectoryStateOnSurface updatedTSOS_next(updatedTSOS);

        // loop over layers compatible with updated TSOS
        for (auto compLayer : compLayers) {
            int nmeas=0;

            if (addtnl_layers_scanned>=max_addtnl_layers) break;    // break if we already looped over enough layers
            if (found_compatible_on_next_layer>0) break;    // break if we already found additional hit

            // find dets compatible with updated TSOS
            std::vector< GeometricSearchDet::DetWithState > dets_next;
            TrajectoryStateOnSurface onLayer_next(updatedTSOS);

            compLayer->compatibleDetsV(onLayer_next, propagatorAlong, *estimator, dets_next);

            //if (!detWithState.size()) continue;
            std::vector<TrajectoryMeasurement> meas_next;

            // find measurements on dets_next and save the valid ones
            for (std::vector<GeometricSearchDet::DetWithState>::iterator idet_next=dets_next.begin(); idet_next!=dets_next.end(); ++idet_next) {
                MeasurementDetWithData det = measurementTracker->idToDet(idet_next->first->geographicalId());

                if (det.isNull()) continue;    // skip if det does not exist
                if (!idet_next->second.isValid()) continue;    // skip if TSOS is invalid

                // Find measurements on this det
                std::vector <TrajectoryMeasurement>mymeas_next=det.fastMeasurements(idet_next->second, onLayer_next, propagatorAlong, *estimator);

                for (std::vector<TrajectoryMeasurement>::const_iterator imea_next=mymeas_next.begin(), ed2=mymeas_next.end(); imea_next != ed2; ++imea_next) {

                    // save valid measurements
                    if (imea_next->recHit()->isValid()) meas_next.push_back(*imea_next);

                }    // end loop over mymeas_next
            }    // end loop over dets_next

            // sort valid measurements found on this layer
            std::sort(meas_next.begin(), meas_next.end(), TrajMeasLessEstim());

            // loop over valid measurements compatible with updated TSOS (TSOS updated with a hit on the first layer)
            for (std::vector<TrajectoryMeasurement>::const_iterator mea_next=meas_next.begin(); mea_next!=meas_next.end(); ++mea_next) {

                if (nmeas>=max_meas) break;    // skip if we already found enough hits

                // try to update TSOS again, with an additional hit
                updatedTSOS_next = updator_->update(mea_next->forwardPredictedState(), *mea_next->recHit());

                if (not updatedTSOS_next.isValid()) continue;    // skip if TSOS updated with additional hit is not valid

                // If there was a compatible hit on this layer, we end up here.
                // An additional compatible hit is saved.
                seedHits.push_back(*mea_next->recHit()->hit());
                det_id = mea_next->recHit()->geographicalId().rawId();
                nmeas++;
                found_compatible_on_next_layer++;

            } // end loop over meas_next

            addtnl_layers_scanned++;    

        } // end loop over compLayers (additional layers scanned after the original layer)

        if (found_compatible_on_next_layer==0) continue;
        // only consider the hit if there was a compatible hit on one of the additional scanned layers

        // Create a seed from two saved hits
        PTrajectoryStateOnDet const& pstate = trajectoryStateTransform::persistentState(updatedTSOS_next, det_id);
        TrajectorySeed seed(pstate, std::move(seedHits), oppositeToMomentum);

        LogTrace("TSGForOIDNN") << "TSGForOIDNN::makeSeedsFromHitDoublets: Number of seedHits: " << seedHits.size() << std::endl;
        out.push_back(seed);

        found++;
        numSeedsMade++;
        hitDoubletSeedsMade++;

        if (found == numOfHitsToTry_) break;    // break if enough measurements scanned
        if (hitDoubletSeedsMade > maxHitDoubletSeeds_) return;    // abort if enough seeds created

    } // end loop over measurements compatible with original TSOS

    if (found)
        layerCount++;

}


//
// Calculate the dynamic error SF by analysing the L2
//
double TSGForOIDNN::calculateSFFromL2(const reco::TrackRef track) const {
    double theSF = 1.0;
    // L2 direction vs pT blowup - as was previously done:
    // Split into 4 pT ranges: <pT1_, pT1_<pT2_, pT2_<pT3_, <pT4_: 13,30,70
    // Split into different eta ranges depending in pT
    double abseta = std::abs(track->eta());
    if (track->pt() <= pT1_)
        theSF = SF1_;
    else if (track->pt() > pT1_ && track->pt() <= pT2_) {
        if (abseta <= eta3_)
            theSF = SF3_;
        else if (abseta > eta3_ && abseta <= eta6_)
            theSF = SF2_;
        else if (abseta > eta6_)
            theSF = SF3_;
    } else if (track->pt() > pT2_ && track->pt() <= pT3_) {
        if (abseta <= eta1_)
            theSF = SF6_;
        else if (abseta > eta1_ && abseta <= eta2_)
            theSF = SF4_;
        else if (abseta > eta2_ && abseta <= eta3_)
            theSF = SF6_;
        else if (abseta > eta3_ && abseta <= eta4_)
            theSF = SF1_;
        else if (abseta > eta4_ && abseta <= eta5_)
            theSF = SF1_;
        else if (abseta > eta5_)
            theSF = SF5_;
    } else if (track->pt() > pT3_) {
        if (abseta <= eta3_)
            theSF = SF5_;
        else if (abseta > eta3_ && abseta <= eta4_)
            theSF = SF4_;
        else if (abseta > eta4_ && abseta <= eta5_)
            theSF = SF4_;
        else if (abseta > eta5_)
            theSF = SF5_;
    }

    LogTrace(theCategory_) << "TSGForOIDNN::calculateSFFromL2: SF has been calculated as: " << theSF;

    return theSF;
}

std::map<std::string, float> TSGForOIDNN::getFeatureMap(
    reco::TrackRef l2,
    const TrajectoryStateOnSurface& tsos_IP,
    const TrajectoryStateOnSurface& tsos_MuS
) const {
    std::map<std::string, float> the_map;
    the_map["pt"] = l2->pt();
    the_map["eta"] = l2->eta();
    the_map["phi"] = l2->phi();
    the_map["validHits"] = l2->found();
    if (tsos_IP.isValid()) {
        the_map["tsos_IP_eta"] = tsos_IP.globalPosition().eta();
        the_map["tsos_IP_phi"] = tsos_IP.globalPosition().phi();
        the_map["tsos_IP_pt"] = tsos_IP.globalMomentum().perp();
        the_map["tsos_IP_pt_eta"] = tsos_IP.globalMomentum().eta();
        the_map["tsos_IP_pt_phi"] = tsos_IP.globalMomentum().phi();
        AlgebraicSymMatrix55 matrix_IP = tsos_IP.curvilinearError().matrix();
        the_map["err0_IP"] = sqrt(matrix_IP[0][0]);
        the_map["err1_IP"] = sqrt(matrix_IP[1][1]);
        the_map["err2_IP"] = sqrt(matrix_IP[2][2]);
        the_map["err3_IP"] = sqrt(matrix_IP[3][3]);
        the_map["err4_IP"] = sqrt(matrix_IP[4][4]);
        the_map["tsos_IP_valid"] = 1.0;
    } else {
        the_map["tsos_IP_eta"] = -999;
        the_map["tsos_IP_phi"] = -999;
        the_map["tsos_IP_pt"] = -999;
        the_map["tsos_IP_pt_eta"] = -999;
        the_map["tsos_IP_pt_phi"] = -999;
        the_map["err0_IP"] = -999;
        the_map["err1_IP"] = -999;
        the_map["err2_IP"] = -999;
        the_map["err3_IP"] = -999;
        the_map["err4_IP"] = -999;
        the_map["tsos_IP_valid"] = 0.0;
    }
    if (tsos_MuS.isValid()) {
        the_map["tsos_MuS_eta"] = tsos_MuS.globalPosition().eta();
        the_map["tsos_MuS_phi"] = tsos_MuS.globalPosition().phi();
        the_map["tsos_MuS_pt"] = tsos_MuS.globalMomentum().perp();
        the_map["tsos_MuS_pt_eta"] = tsos_MuS.globalMomentum().eta();
        the_map["tsos_MuS_pt_phi"] = tsos_MuS.globalMomentum().phi();
        AlgebraicSymMatrix55 matrix_MuS = tsos_MuS.curvilinearError().matrix();
        the_map["err0_MuS"] = sqrt(matrix_MuS[0][0]);
        the_map["err1_MuS"] = sqrt(matrix_MuS[1][1]);
        the_map["err2_MuS"] = sqrt(matrix_MuS[2][2]);
        the_map["err3_MuS"] = sqrt(matrix_MuS[3][3]);
        the_map["err4_MuS"] = sqrt(matrix_MuS[4][4]);
        the_map["tsos_MuS_valid"] = 1.0;
    } else {
        the_map["tsos_MuS_eta"] = -999;
        the_map["tsos_MuS_phi"] = -999;
        the_map["tsos_MuS_pt"] = -999;
        the_map["tsos_MuS_pt_eta"] = -999;
        the_map["tsos_MuS_pt_phi"] = -999;
        the_map["err0_MuS"] = -999;
        the_map["err1_MuS"] = -999;
        the_map["err2_MuS"] = -999;
        the_map["err3_MuS"] = -999;
        the_map["err4_MuS"] = -999;
        the_map["tsos_MuS_valid"] = 0.0;
    }
    return the_map;
}


std::tuple<int, int, int, bool> TSGForOIDNN::evaluateDnn(
    std::map<std::string, float> feature_map,
    tensorflow::Session* session,
    const pt::ptree& metadata
) const {
    int nHB, nHLIP,nHLMuS, n_features = 0;
    bool dnnSuccess = false;
    n_features = metadata.get<int>("n_features", 0);

    // Prepare tensor for DNN inputs
    tensorflow::Tensor input(tensorflow::DT_FLOAT, { 1, n_features });
    std::string fname;
    int i_feature = 0;
    for (const pt::ptree::value_type &feature : metadata.get_child("feature_names")){
        fname =  feature.second.data();
        if (feature_map.find(fname) == feature_map.end()) {
            std::cout << "Couldn't find " << fname << " in feature_map! Will not evaluate DNN." << std::endl;
            return std::make_tuple(nHB, nHLIP, nHLMuS, dnnSuccess);
        }
        else {
            input.matrix<float>()(0, i_feature) = float(feature_map.at(fname));
            i_feature++;
        }
    }

    // Prepare tensor for DNN outputs
    std::vector<tensorflow::Tensor> outputs;

    // Evaluate DNN and put results in output tensor
    std::string inputLayer = metadata.get<std::string>("input_layer");
    std::string outputLayer = metadata.get<std::string>("output_layer");
    //std::cout << inputLayer << " " << outputLayer << std::endl;
    tensorflow::run(session, { { inputLayer, input } }, { outputLayer }, &outputs);
    tensorflow::Tensor out_tensor = outputs[0];
    tensorflow::TTypes<float, 1>::Matrix dnn_outputs = out_tensor.matrix<float>();

    // Find output with largest prediction
    int imax = -1;
    float out_max = 0;
    for (long long int i = 0; i < out_tensor.dim_size(1); i++) {
        float ith_output = dnn_outputs(0, i);
        if (ith_output > out_max){
            imax = i;
            out_max = ith_output;
        }
    }

    // Decode output
    nHB = metadata.get<int>("output_labels.label_"+std::to_string(imax+1)+".nHB");
    nHLIP = metadata.get<int>("output_labels.label_"+std::to_string(imax+1)+".nHLIP");
    nHLMuS = metadata.get<int>("output_labels.label_"+std::to_string(imax+1)+".nHLMuS");

    dnnSuccess = true;
    return std::make_tuple(nHB, nHLIP, nHLMuS, dnnSuccess);
}


//
//
//
void TSGForOIDNN::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltL2Muons", "UpdatedAtVtx"));
  desc.add<int>("layersToTry", 2);
  desc.add<double>("fixedErrorRescaleFactorForHitless", 2.0);
  desc.add<int>("hitsToTry", 1);
  desc.add<bool>("adjustErrorsDynamicallyForHitless", true);
  desc.add<edm::InputTag>("MeasurementTrackerEvent", edm::InputTag("hltSiStripClusters"));
  desc.add<std::string>("estimator", "hltESPChi2MeasurementEstimator100");
  desc.add<double>("maxEtaForTOB", 1.8);
  desc.add<double>("minEtaForTEC", 0.7);
  desc.addUntracked<bool>("debug", false);
  desc.add<unsigned int>("maxSeeds", 20);
  desc.add<unsigned int>("maxHitlessSeeds", 5);  
  desc.add<unsigned int>("maxHitSeeds", 1);
  desc.add<double>("pT1", 13.0);
  desc.add<double>("pT2", 30.0);
  desc.add<double>("pT3", 70.0);
  desc.add<double>("eta1", 0.2);
  desc.add<double>("eta2", 0.3);
  desc.add<double>("eta3", 1.0);
  desc.add<double>("eta4", 1.2);
  desc.add<double>("eta5", 1.6);
  desc.add<double>("eta6", 1.4);
  desc.add<double>("eta7", 2.1);
  desc.add<double>("SF1", 3.0);
  desc.add<double>("SF2", 4.0);
  desc.add<double>("SF3", 5.0);
  desc.add<double>("SF4", 7.0);
  desc.add<double>("SF5", 10.0);
  desc.add<double>("SF6", 2.0);
  desc.add<std::string>("propagatorName", "PropagatorWithMaterialParabolicMf");
  desc.add<unsigned int>("maxHitlessSeedsIP", 5);
  desc.add<unsigned int>("maxHitlessSeedsMuS", 0);
  desc.add<unsigned int>("maxHitDoubletSeeds", 0);
  desc.add<bool>("getStrategyFromDNN", false);
  desc.add<double>("etaSplitForDnn", 1.0);
  desc.add<std::string>("dnnMetadataPath", "");
  descriptions.add("TSGForOIDNN", desc);
}

DEFINE_FWK_MODULE(TSGForOIDNN);
