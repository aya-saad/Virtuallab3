from django.contrib import admin

from .models import *

admin.site.register(Experiment)
admin.site.register(Download)
admin.site.register(ExpDownload)
admin.site.register(XML)
admin.site.register(ExpXML)
admin.site.register(PDF)
admin.site.register(ExpPDF)
admin.site.register(Video)
admin.site.register(ExpVideo)
