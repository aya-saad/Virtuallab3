from django.shortcuts import render

# Create your views here.
import os
from django.conf import settings
from django.http import HttpResponse
from django.template import loader
from django.views import generic
from django.shortcuts import get_object_or_404, render
from models.models import Experiment

def main_page(request):
    return render(request, 'home.html')


def models(request):
    return render(request, 'models.html')


def simulate(request):
    return render(request, 'fmulab/simulate.html')


def downloads(request):
    template = loader.get_template('downloads.html')
    exp_list = Experiment.objects.order_by('exp_num')[:]
    context = {
        'exp_list': exp_list,
    }
    return HttpResponse(template.render(context, request))

