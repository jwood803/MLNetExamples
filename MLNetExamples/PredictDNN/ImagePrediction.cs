using System;
using System.Collections.Generic;
using System.Text;

namespace PredictDNN
{
    public class ImagePrediction
    {
        public float[] Score { get; set; }

        public uint PredictedLabel { get; set; }
    }
}
