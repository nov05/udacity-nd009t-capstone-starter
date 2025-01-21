* Start the container  

```cmd
docker run -it ^
-v "C:\Users\guido\.aws":/home/glue_user/.aws ^
-v "D:\github\udacity-nd009t-capstone-starter\docker_workspace":/home/glue_user/workspace/ ^
-e AWS_PROFILE="admin" ^
-e DISABLE_SSL=true ^
--rm -p 4040:4040 -p 18080:18080 ^
--name aws_glue_pyspark ^
amazon/aws-glue-libs:glue_libs_4.0.0_image_01 pyspark
```