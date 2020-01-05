#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

class SeedCandidate {
  public:
    SeedCandidate() {}
    SeedCandidate(const reco::Track& l2_, const std::string layerId_, int layerNum_, const std::string seedType_){
      pt = l2_.pt();
      eta = l2_.eta();
      phi = l2_.phi();
      layerId = layerId_;
      layerNum = layerNum_;
      seedType = seedType_;
    }
    double pt;
    double eta;
    double phi;
    std::string layerId;
    int layerNum;
    std::string seedType;
};
