#!/bin/bash
pushd $(dirname $0) > /dev/null
chmod +x gdrivedl
./gdrivedl https://drive.google.com/open?id=1anOEzUMxXR4ff2qaUJNojAABWuAmaGvw
./gdrivedl https://drive.google.com/open?id=1t21DG1ts-2NQXDVvuQjW9LY9VVkYuXU5
./gdrivedl https://drive.google.com/open?id=1KFg7ZjdJyhLE58WzEBlznOrDpKmQqviC
./gdrivedl https://drive.google.com/open?id=1GqaB80sznVkonprCdYhURwGwqiPRhP-v
./gdrivedl https://drive.google.com/open?id=1gRFn2NojH47BjURSws2XIyuTjzFkmuSV
popd > /dev/null
