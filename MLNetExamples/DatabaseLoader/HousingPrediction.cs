using Microsoft.ML.Data;

namespace DbLoader
{
    public class HousingPrediction
    {
        [ColumnName("Score")]
        public float PredictedHouseValue { get; set; }
    }
}
