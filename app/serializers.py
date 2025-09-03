from rest_framework import serializers
from .models import Factcheck

class FactcheckSerializer(serializers.ModelSerializer):
    class Meta:
        model = Factcheck
        fields = '__all__'
