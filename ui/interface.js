var original_image;

$(document).ready(function (e) {
  $('#upload').on('click', function () {
    var form_data = new FormData();
    var ins = document.getElementById('file').files.length;

    if(ins == 0) {
      $('#msg').html('<span style="color:red">Select at least one file</span>');
      return;
    }

    for (var x = 0; x < ins; x++) {
      form_data.append("files[]", document.getElementById('file').files[x]);
    }
    form_data.append("nms_thresh", document.getElementById("nms").value)
    form_data.append("conf_thresh", document.getElementById("conf").value)
    form_data.append("device", document.getElementById("device").value)

    $.ajax({
      url: 'http://127.0.0.1:5000/process_image', // point to server-side URL
      dataType: 'json', // what to expect back from server
      cache: false,
      contentType: false,
      processData: false,
      data: form_data,
      type: 'post',
      success: function (response) { // display success response
        $('#time_taken').html(response['time_taken']);
        console.log(response)
        draw(response['data']);
        $.each(response, function (key, data) {
          if(key !== 'message') {
            $('#msg').append(key + ' -> ' + data + '<br/>');
          } else {
            $('#msg').append(data + '<br/>');
          }
        })
      },
      error: function (response) {
        $('#msg').html(response.message); // display error response
      }
    });
  });
});


var loadFile = function(event) {
  var image_holder = document.getElementById('input');
  original_image = event.target.files[0]
  image_holder.src = URL.createObjectURL(original_image);
};


function draw(boxes){
    boxes = boxes['boxes']
    $('#nr_detections').html(boxes.length)
    console.log(boxes)
    var original_image = document.getElementById("input");
    width = original_image.width;
    height = original_image.height;

    var canvas = document.getElementById('drawing');
    var ctx = canvas.getContext('2d');

    ctx.canvas.width = width;
    ctx.canvas.height = height;

    console.log(width, height)

    ctx.drawImage(original_image, 0, 0);

    ctx.beginPath();
    ctx.strokeStyle = 'rgb(0, 255, 0)';
    for (box of boxes){
      console.log(box);
      ctx.rect(box[0], box[1], box[2], box[3]);
    }
    ctx.stroke();
}
