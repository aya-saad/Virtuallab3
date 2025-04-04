from django.db import models


class Experiment(models.Model):
    """Experiment Number
    Title
    Description
    Downloads
    Kopl & osp.xml
    Textual explanation (downloadable pdf instruction)
    Video explanation """
    # fields of the model
    exp_num = models.IntegerField()
    exp_title = models.TextField(max_length=100)
    exp_desc = models.TextField(max_length=3000)

    # exp_downloads = models.TextField(max_length=1000)
    # exp_xml = models.TextField(max_length=1000)
    # exp_pdf = models.TextField(max_length=1000)
    # exp_video = models.TextField(max_length=1000)
    # renames the instances of the model
    # with their title name
    def __str__(self):
        return self.exp_title


class Download(models.Model):
    # fields of the model
    download_num = models.IntegerField()
    download_title = models.TextField(max_length=100)
    download_link = models.TextField(max_length=500)
    download_desc = models.TextField(max_length=3000)

    def __str__(self):
        return self.download_title


class ExpDownload(models.Model):
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)
    download = models.ForeignKey(Download, on_delete=models.CASCADE)
    exp_dload = models.TextField(max_length=500)

    def __str__(self):
        return self.exp_dload


class XML(models.Model):
    # fields of the model
    xml_num = models.IntegerField()
    xml_title = models.TextField(max_length=100)
    xml_link = models.TextField(max_length=500)
    xml_desc = models.TextField(max_length=3000)

    def __str__(self):
        return self.xml_title


class ExpXML(models.Model):
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)
    xml = models.ForeignKey(XML, on_delete=models.CASCADE)
    exp_xml = models.CharField(max_length=1000)

    def __str__(self):
        return self.exp_xml


class PDF(models.Model):
    # fields of the model
    pdf_num = models.IntegerField()
    pdf_title = models.TextField(max_length=100)
    pdf_link = models.TextField(max_length=500)
    pdf_desc = models.TextField(max_length=3000)

    def __str__(self):
        return self.pdf_title


class ExpPDF(models.Model):
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)
    pdf = models.ForeignKey(PDF, on_delete=models.CASCADE)
    exp_pdf = models.CharField(max_length=1000)

    def __str__(self):
        return self.exp_pdf


class Video(models.Model):
    # fields of the model
    video_num = models.IntegerField()
    video_title = models.TextField(max_length=100)
    video_link = models.TextField(max_length=500)
    video_desc = models.TextField(max_length=3000)

    def __str__(self):
        return self.video_title


class ExpVideo(models.Model):
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    exp_video = models.CharField(max_length=1000)

    def __str__(self):
        return self.exp_video
