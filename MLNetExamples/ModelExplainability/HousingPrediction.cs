using Microsoft.ML.Data;

namespace ModelExplainability
{
    public class HousingPrediction
    {
        [ColumnName("Score")]
        public float PredictedHouseValue { get; set; }

        public float[] FeatureContributions { get; set; }
    }
}
