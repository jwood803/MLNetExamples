using Microsoft.ML.Data;

namespace LargeFeatures
{
    public class SensorPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction;

        public float Score;
    }
}
