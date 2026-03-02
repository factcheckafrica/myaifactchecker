from rest_framework import serializers
from .models import Factcheck

class FactcheckSerializer(serializers.ModelSerializer):
    class Meta:
        model = Factcheck
        fields = '__all__'


# myapp/serializers.py
from rest_framework import serializers

class FactcheckRequestSerializer(serializers.Serializer):
    query = serializers.CharField(max_length=1000)

class CitationSerializer(serializers.Serializer):
    title = serializers.CharField()
    url = serializers.URLField()
    tier = serializers.CharField(required=False)

class FactcheckResultJsonSerializer(serializers.Serializer):
    verdict = serializers.CharField()
    explanation = serializers.CharField()
    citations = CitationSerializer(many=True, required=False)
    confidence = serializers.FloatField(required=False)

class FactcheckResponseSerializer(serializers.Serializer):
    query = serializers.CharField()
    result = serializers.CharField(required=False, allow_blank=True)
    result_json = FactcheckResultJsonSerializer(required=False)
    latency_ms = serializers.IntegerField()

