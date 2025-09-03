from django.db import models
from django.contrib.auth.models import User
from autoslug import AutoSlugField

class Factcheck(models.Model):
    user_input_news = models.TextField(max_length=1000, null=True , blank=True)
    fresult = models.TextField(max_length=1000, null=True , blank=True)
    gpt_check_result = models.TextField(max_length=1000, null=True , blank=True)
    sentiment_label = models.CharField(max_length=1000, null=True , blank=True)
    num_genuine_sources = models.IntegerField(default=0)
    non_authentic_sources = models.IntegerField(default=0)
    genuine_urls = models.JSONField(default=list, null=True)
    non_authentic_urls = models.JSONField(default=list, null=True)
    genuine_urls_and_dates = models.JSONField(default=dict, null=True)
    slug=AutoSlugField(populate_from='user_input_news',null=True, blank=True, unique=True)
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    
    def __str__(self):
        return self.user_input_news
    
class ActiveUser(models.Model):
    user = models.ForeignKey(User, null=True, on_delete=models.DO_NOTHING)
    ip_address = models.PositiveBigIntegerField()

class UserReport(models.Model):
    name = models.CharField(max_length=255)
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} - {self.timestamp}"
    


