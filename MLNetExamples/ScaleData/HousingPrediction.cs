using Microsoft.ML.Data;

namespace ScaleData
{
    public class HousingPrediction
    {
        [ColumnName("Score")]
        public float PredictedHouseValue { get; set; }
    }
}
