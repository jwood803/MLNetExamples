using Microsoft.ML.Data;

namespace FeatureImportance
{
    public class HousingPrediction
    {
        [ColumnName("Score")]
        public float PredictedHouseValue { get; set; }
    }
}
