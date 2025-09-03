from rest_framework import serializers
from .models import Factcheck, ActiveUser, UserReport
from django.contrib.auth.models import User


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name']


class FactcheckSerializer(serializers.ModelSerializer):
    class Meta:
        model = Factcheck
        fields = [
            'id', 'user_input_news', 'fresult', 'gpt_check_result',
            'sentiment_label', 'num_genuine_sources', 'non_authentic_sources',
            'genuine_urls', 'non_authentic_urls', 'genuine_urls_and_dates',
            'slug', 'created_at'
        ]
        read_only_fields = ['id', 'slug', 'created_at']


class FactcheckCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Factcheck
        fields = ['user_input_news']
        
    def validate_user_input_news(self):
        value = self.validated_data.get('user_input_news')
        if not value or len(value.strip()) < 10:
            raise serializers.ValidationError("Please provide a meaningful news statement to fact-check.")
        return value


class ActiveUserSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    
    class Meta:
        model = ActiveUser
        fields = ['id', 'user', 'ip_address']


class UserReportSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserReport
        fields = ['id', 'name', 'message', 'timestamp']
        read_only_fields = ['id', 'timestamp']
