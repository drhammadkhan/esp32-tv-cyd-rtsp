#pragma once

#include "ChannelData.h"
#include <vector>
#include <string>

class NetworkChannelData : public ChannelData
{
private:
  std::string mChannelInfoURL;
  std::vector<int> mChannelLengths;
  std::string mFrameURL;
  std::string mAudioURL;
  std::string mClientQuery;
public:
  NetworkChannelData(const char *channelInfoURL, const char *frameURL, const char *audioURL, const char *clientQuery);
  bool fetchChannelData();
  int getChannelCount() {
    return mChannelLengths.size();
  }
  int getChannelLength(int channelIndex) {
    return mChannelLengths[channelIndex];
  }
  std::string getFrameURL();
  std::string getAudioURL();
  std::string getClientQuery();
  void setChannel(int channel);
};
