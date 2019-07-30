using Microsoft.ML.Data;

namespace AnomalyDetection
{
    public class EnergyPrediction
    {
        [VectorType(2)]
        public double[] Prediction { get; set; }
    }
}
