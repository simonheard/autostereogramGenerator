def process_request(request):
    """
    Responds to an HTTP request using data in the request object.
    Supports GET and POST methods.
    
    Args:
        request (flask.Request): HTTP request object.

    Returns:
        str: Response message or data.
    """
    # Check the HTTP method
    if request.method == 'GET':
        # Get parameters from the query string
        name = request.args.get('name', 'World')
        return f'Hello, {name}!'

    elif request.method == 'POST':
        # Get data from the posted JSON
        request_json = request.get_json(silent=True)
        name = request_json.get('name', 'World') if request_json else 'World'
        return f'Hello, {name}!', 200

    else:
        # Other HTTP methods not supported
        return 'Unsupported method', 405
