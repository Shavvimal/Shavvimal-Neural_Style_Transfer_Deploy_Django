from django.shortcuts import render
from .apps import WebappConfig 

# Create your views here.
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .apps import WebappConfig

import tensorflow as tf
import base64



class call_model(APIView):

    def get(self,request):
        if request.method == 'GET':
            
            # sentence is the query we want to get the prediction for
            content =  request.GET.get('content')
            style =  request.GET.get('style')

            content_path = tf.keras.utils.get_file('content.jpg', content)
            style_path = tf.keras.utils.get_file('style.jpg', style)
            
            content_image = WebappConfig.load_img(content_path)
            style_image = WebappConfig.load_img(style_path)


            # predict method used to get the prediction
            stylized_image = WebappConfig.model(tf.constant(content_image), tf.constant(style_image))[0]
            
            # returning JSON response
            final_image=WebappConfig.tensor_to_image(stylized_image)
            # encoded_image=tf.io.encode_base64(stylized_image)

            response = HttpResponse(content_type='image/jpg')
            final_image.save(response, "JPEG")
            response['Content-Disposition'] = 'attachment; filename="stylized_image.jpg"'

            tf.io.gfile.remove(content_path)
            tf.io.gfile.remove(style_path)


            return response




