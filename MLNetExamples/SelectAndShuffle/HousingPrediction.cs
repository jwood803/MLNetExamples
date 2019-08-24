using Microsoft.ML.Data;

namespace SelectAndShuffle
{
    public class HousingPrediction
    {
        [ColumnName("Score")]
        public float PredictedPrice { get; set; }
    }
}
