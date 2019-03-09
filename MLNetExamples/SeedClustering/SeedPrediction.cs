using Microsoft.ML.Data;

namespace SeedClustering
{
    public class SeedPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint SelectedClusterId;
        [ColumnName("Score")]
        public float[] Distance;
    }
}
