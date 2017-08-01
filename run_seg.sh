ROOT=/home/chenqi/places_challenge_2017/sceneparsing

mrun="/home/chenqi/bin/MATLAB/R2017a/bin/matlab -nodesktop -nodisplay -nosplash -r"

$mrun "demoSegmentation $ROOT/models/deploy_DilatedNet.prototxt \
                        $ROOT/trainingCode/caffe/snapshot/dilatedNet_iter_50000.caffemodel \
                        dilatednet_10w"
                        # $ROOT/models/DilatedNet_iter_120000.caffemodel \
