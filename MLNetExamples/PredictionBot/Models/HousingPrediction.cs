using Microsoft.ML.Data;

namespace PredictionBot
{
    public class HousingPrediction
    {
        [ColumnName("Score")]
        public float PredictedHouseValue { get; set; }
    }
}
