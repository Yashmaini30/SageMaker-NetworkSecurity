provider "aws" {
  region = "us-east-1" 
}

resource "aws_s3_bucket" "sagemaker_bucket" {
  bucket = "threat-matrix-sagemaker-bucket"  

  tags = {
    Name        = "SageMaker Bucket"
    Environment = "Dev"
  }
}

#versioning
resource "aws_s3_bucket_versioning" "versioning" {
  bucket = aws_s3_bucket.sagemaker_bucket.id

  versioning_configuration {
    status = "Enabled"
  }
}

# blocking access
resource "aws_s3_bucket_public_access_block" "public_access" {
  bucket = aws_s3_bucket.sagemaker_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
