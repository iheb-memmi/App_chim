package com.bcxp.whatcar;

import static android.content.ContentValues.TAG;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;

import com.bcxp.whatcar.ml.MyModel;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.material.snackbar.Snackbar;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.provider.MediaStore;
import android.util.Log;
import android.view.View;

import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;

import com.bcxp.whatcar.databinding.ActivityMainBinding;
import com.google.firebase.ml.modeldownloader.CustomModel;
import com.google.firebase.ml.modeldownloader.CustomModelDownloadConditions;
import com.google.firebase.ml.modeldownloader.DownloadType;
import com.google.firebase.ml.modeldownloader.FirebaseModelDownloader;

import android.view.Menu;
import android.view.MenuItem;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.common.ops.QuantizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    private AppBarConfiguration appBarConfiguration;
    private ActivityMainBinding binding;

    Button  camera;
    ImageButton gallery;
    TextView result;
    ImageView imageView;

    int  imgSize = 448;
    private Interpreter interpreter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        gallery = findViewById(R.id.button2);
        camera = findViewById(R.id.button);

        imageView = findViewById(R.id.imageView);
        result = findViewById(R.id.result);


        gallery.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View view) {
                Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraIntent, 1);
            }
        });
        camera.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                } else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
    }

    public void classifyImage(Bitmap image) throws IOException {

        //MyModel model = MyModel.newInstance(getApplicationContext());
        FileInputStream f_input_stream= new FileInputStream(new File("ml/my_model.tflite"));
        FileChannel f_channel = f_input_stream.getChannel();
        MappedByteBuffer tflite_model = f_channel.map(FileChannel.MapMode.READ_ONLY, 0, f_channel .size());

        //File modelFile = new File("ml/my_model.tflite");
        interpreter = new Interpreter(tflite_model);

        Bitmap bitmap = Bitmap.createScaledBitmap(image, 448, 448, true);
        ByteBuffer input = ByteBuffer.allocateDirect(3 * 448 * 448 * 4).order(ByteOrder.nativeOrder());
        for (int y = 0; y < 448; y++) {
            for (int x = 0; x < 448; x++) {
                int px = bitmap.getPixel(x, y);

                // Get channel values from the pixel value.
                int r = Color.red(px);
                int g = Color.green(px);
                int b = Color.blue(px);

                // Normalize channel values to [-1.0, 1.0]. This requirement depends
                // on the model. For example, some models might require values to be
                // normalized to the range [0.0, 1.0] instead.
                float rf = r  / 255.0f;
                float gf = g  / 255.0f;
                float bf = b  / 255.0f;

                input.putFloat(rf);
                input.putFloat(gf);
                input.putFloat(bf);
            }
        }

        // output buffer
        int bufferSize = 15 * java.lang.Float.SIZE / java.lang.Byte.SIZE;
        ByteBuffer modelOutput = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());
        interpreter.run(input, modelOutput);

        // read output
        String[] labels = {"A205", "A207", "A238", "C204", "C205", "C207", "C238", "S204", "S205", "S212", "S213", "W204", "W205", "W212", "W213"};

        modelOutput.rewind();
        FloatBuffer probabilities = modelOutput.asFloatBuffer();
        for (int i = 0; i < probabilities.capacity(); i++) {
            String label = labels[i];
            float probability = probabilities.get(i);
            Log.i(TAG, String.format("%s: %1.4f", label, probability));
        }
    }


    public void classifyImage_alt(Bitmap image){
        try {
            MyModel model = MyModel.newInstance(getApplicationContext());


            // Initialization code
            // Create an ImageProcessor with all ops required. For more ops, please
            // refer to the ImageProcessor Architecture section in this README.
            ImageProcessor imageProcessor =
                    new ImageProcessor.Builder()
                            .add(new ResizeOp(448, 448, ResizeOp.ResizeMethod.BILINEAR))
                            .add(new NormalizeOp(127.5F, 127.5F))
                            .add(new QuantizeOp(128.0F, (float) (1/128.0)))
                            .build();

            // Create a TensorImage object. This creates the tensor of the corresponding
            // tensor type (uint8 in this case) that the TensorFlow Lite interpreter needs.
            TensorImage tensorImage = new TensorImage(DataType.FLOAT32);

            // Analysis code for every frame
            // Preprocess the image
            tensorImage.load(image);
            tensorImage = imageProcessor.process(tensorImage);

            // Create a container for the result and specify that this is a quantized model.
            // Hence, the 'DataType' is defined as UINT8 (8-bit unsigned integer)
            TensorBuffer probabilityBuffer =
                    TensorBuffer.createFixedSize(new int[]{1, 15}, DataType.FLOAT32);



            // Initialise the model
           /* Interpreter tflite = null; try {

                tflite = new Interpreter(tensorImage);
            } catch (IOException e) {
                Log.e("tfliteSupport", "Error reading model", e);
            }

            // Running inference
            if (null != tflite) {
                tflite.run(tensorImage.getBuffer(), probabilityBuffer.getBuffer());
            }*/
            // Creates inputs for reference.
/*            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 3, 448, 448}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imgSize * imgSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imgSize * imgSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
            for(int i = 0; i < imgSize; i ++){
                for(int j = 0; j < imgSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 1));
                }
            }
            */
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 3, 448, 448}, DataType.FLOAT32);
            inputFeature0.loadBuffer(tensorImage.getBuffer());


            // Runs model inference and gets result.
            MyModel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            float[] data = outputFeature0.getFloatArray() ;
            Log.d(TAG," Heeeeeeeeeeeeeeere :");
            Log.d(TAG, String.valueOf(data));
            //System.out.println(outputFeature0);

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            // int maxPos = 0 ;
            String[] labels = {"A205", "A207", "A238", "C204", "C205", "C207", "C238", "S204", "S205", "S212", "S213", "W204", "W205", "W212", "W213"};
            result.setText(labels[maxPos]);

            // Releases model resources if no longer used.
            //model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(resultCode == RESULT_OK){
            if(requestCode == 1){
                Uri dat = data.getData();
                Bitmap image = null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imgSize, imgSize, false); // resize our input image to the size our model requires
                classifyImage(image);
            }else{
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension); // transform the input image to a square
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imgSize, imgSize, false);
                classifyImage(image);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }




    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }


}