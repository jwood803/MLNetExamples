using Microsoft.ML.Data;

namespace EventHubPredict
{
    public class HousingPrediction
    {
        [ColumnName("Score")]
        public float PredictedHouseValue { get; set; }
    }
}
