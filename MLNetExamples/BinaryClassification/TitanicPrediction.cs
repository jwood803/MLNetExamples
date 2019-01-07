using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace BinaryClassification
{
    public class TitanicPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction;

        public float Score;
    }
}
