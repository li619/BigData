package MatrixMultiply;

import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.util.GenericOptionsParser;

public class MatrixMultiply {
    public static final String CONTROL_I = "\u0009";  // Tab character
    public static final int MATRIX_AM = 4;  // 矩阵A的行数
    public static final int MATRIX_ABN = 3; // 矩阵A的列数和矩阵B的行数
    public static final int MATRIX_BL = 2;  // 矩阵B的列数

    // Mapper
    public static class MatrixMapper extends Mapper<LongWritable, Text, Text, Text> {
        private String file;

        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            file = fileSplit.getPath().getName();

            String line = value.toString();
            if (line == null || line.isEmpty()) return;

            String[] values = line.split(" ");
            if (values.length < 3) return;

            String row = values[0];
            String col = values[1];
            String val = values[2];

            if (file.equals("a.txt")) {  // 对应矩阵 A
                for (int i = 1; i <= MATRIX_BL; i++) {
                    context.write(new Text(row + CONTROL_I + i), new Text("a#" + col + "#" + val));
                }
            } else if (file.equals("b.txt")) {  // 对应矩阵 B
                for (int i = 1; i <= MATRIX_AM; i++) {
                    context.write(new Text(i + CONTROL_I + col), new Text("b#" + row + "#" + val));
                }
            }
        }
    }

    // Reducer
    public static class MatrixReducer extends Reducer<Text, Text, Text, Text> {
        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            int[] valA = new int[MATRIX_ABN]; // 矩阵A的列数
            int[] valB = new int[MATRIX_ABN]; // 矩阵B的行数
            for (int i = 0; i < MATRIX_ABN; i++) {
                valA[i] = 0;
                valB[i] = 0;
            }

            for (Text value : values) {
                String val = value.toString();
                if (val.startsWith("a#")) {
                    String[] parts = val.split("#");
                    int colIndex = Integer.parseInt(parts[1]) - 1;
                    valA[colIndex] = Integer.parseInt(parts[2]);
                } else if (val.startsWith("b#")) {
                    String[] parts = val.split("#");
                    int rowIndex = Integer.parseInt(parts[1]) - 1;
                    valB[rowIndex] = Integer.parseInt(parts[2]);
                }
            }

            int result = 0;
            for (int i = 0; i < MATRIX_ABN; i++) {
                result += valA[i] * valB[i];
            }
            context.write(key, new Text(Integer.toString(result)));  // 输出矩阵C的元素
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();

        if (otherArgs.length != 2) {
            System.err.println("Usage: MatrixMultiply <in> <out>");
            System.exit(2);
        }

        // 确保输出目录不存在
        Path outputPath = new Path(otherArgs[1]);
        FileSystem fs = outputPath.getFileSystem(conf);
        if (fs.exists(outputPath)) {
            fs.delete(outputPath, true);  // 删除已存在的输出路径
        }

        Job job = Job.getInstance(conf, "MatrixMultiply");
        job.setJarByClass(MatrixMultiply.class);
        job.setMapperClass(MatrixMapper.class);
        job.setReducerClass(MatrixReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
        FileOutputFormat.setOutputPath(job, outputPath);

        boolean success = job.waitForCompletion(true);
        if (success) {
            System.out.println("Matrix multiplication completed successfully.");
            System.exit(0);
        } else {
            System.out.println("Matrix multiplication failed.");
            System.exit(1);
        }
    }
}
