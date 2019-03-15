from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt


def _handle_uploaded_file(image):
    '''
        This function processes the image and return the
        download link to the images if the images belongs to a
        cluster
    '''
    # code for processing goes here
    pass


@csrf_exempt
def get_images(request):
    '''
        Handles requests for getting images.
        On a POST request returns a link to the images
    '''
    if request.method == 'POST':
        if request.FILES.get('image'):
            # link = _handle_uploaded_file(request.FILES.get('image'))
            return JsonResponse({"link": "#"})
        else:
            return JsonResponse({"message": "not found"}, status=500)
    else:
        return JsonResponse({"message": "not found"}, status=404)
