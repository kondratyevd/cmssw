#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

class SeedCandidate {
  public:
    SeedCandidate() {}
    SeedCandidate(int l2_idx_, const reco::Track& l2_, const std::string layerId_, int layerNum_, const std::string seedType_){
      l2_idx = l2_idx_;
      pt = l2_.pt();
      eta = l2_.eta();
      phi = l2_.phi();
      layerId = layerId_;
      layerNum = layerNum_;
      seedType = seedType_;
    }
    int l2_idx;
    double pt;
    double eta;
    double phi;
    std::string layerId;
    int layerNum;
    std::string seedType;
};
