using Microsoft.ML.Data;

namespace BinaryData
{
    public class HousingPrediction
    {
        [ColumnName("Score")]
        public float PredictedPrice { get; set; }
    }
}
