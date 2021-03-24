if (Array.prototype.equals)
    console.warn("Overriding existing Array.prototype.equals. Possible causes: New API defines the method, there's a framework conflict or you've got double inclusions in your code.");
Array.prototype.equals = function (array) {
    if (!array)
        return false;

    if (this.length != array.length)
        return false;

    for (var i = 0, l=this.length; i < l; i++) {
        if (this[i] instanceof Array && array[i] instanceof Array) {
            if (!this[i].equals(array[i]))
                return false;       
        }           
        else if (this[i] != array[i]) {
            return false;   
        }           
    }       
    return true;
}
Object.defineProperty(Array.prototype, "equals", {enumerable: false});

$(document).ready(function() {
    var labels_prev = [];

    var vertices = [];
    var triangles = [];

    var filename = "mesh.obj";

    $(".form-range").val(0);
    $(".overlay").hide();

    $("#btn-generate").click(function() {
        var labels = $("#chair-labels select").map(function() {
            return $(this).val();
        }).get();

        if (!labels.equals(labels_prev)) {
            $(".form-range").val(0);
        }

        labels_prev = labels;

        var samples = $("#chair-samples input").map(function() {
            return $(this).val();
        }).get();

        console.log(camera)

        $(".overlay").show();
        $(".form-range").prop("disabled", true);

        $.ajax({
            type: "POST",
            url: "/generate",
            data: JSON.stringify({"labels": labels, "samples": samples}),
            contentType: "application/json;charset=UTF-8",
            dataType : "json",
            cache: false,
            processData: false,
            success: function(data) {
                update_sliders(data["sigma"]);
                vertices = data["vertices"];
                triangles = data["triangles"];
                create_scene(vertices, triangles);
                $(".overlay").hide();
            },
        });
    });
    
    $("#btn-export").click(function() {
        if (vertices.length > 0 && triangles.length > 0) {
            exportFile(writeOBJ(vertices, triangles));
        } else {
            alert("The mesh is empty!");
        }
    })

    function update_sliders(sigma) {
        for (i = 0; i < sigma.length; i++) {
            if (i < $("#chair-samples input").length) {
                $("#range" + i).attr("min", -sigma[i]);
                $("#range" + i).attr("max", sigma[i]);
                $("#range" + i).prop("disabled", false);
            }
        }
    }

    var camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);

    var ambient = new THREE.AmbientLight(0xffffff, 0.35);
    camera.add(ambient);

    var point = new THREE.PointLight(0xffffff);
    point.position.set(2, 20, 15);
    camera.add(point);

    camera.position.z = 60;

    function create_scene(vertices, triangles) {
        if ($("canvas").length > 0) {
            $("canvas").remove();
        }

        var renderer = new THREE.WebGLRenderer();
        var container = $("#scene")[0];
        var w = container.offsetWidth;
        var h = container.offsetHeight;
        renderer.setSize(w, h);
        container.append(renderer.domElement);

        var scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf8f9fa);

        scene.add(camera);

        var controls = new THREE.TrackballControls(camera, renderer.domElement);
        controls.rotateSpeed = 5.0;

        var geometry = new THREE.Geometry();

        for (var coord of vertices) {
            var vertex = new THREE.Vector3(coord[0], coord[1], coord[2]);
            geometry.vertices.push(vertex);
        }

        for (var triangle of triangles) {
            var face = new THREE.Face3(triangle[0], triangle[1], triangle[2]);
            geometry.faces.push(face);
        }

        geometry.computeFaceNormals();

        var middle = new THREE.Vector3();

        geometry.computeBoundingBox();

        middle.x = (geometry.boundingBox.max.x + geometry.boundingBox.min.x) / 2;
        middle.y = (geometry.boundingBox.max.y + geometry.boundingBox.min.y) / 2;
        middle.z = (geometry.boundingBox.max.z + geometry.boundingBox.min.z) / 2;

        for (var vertex of geometry.vertices) {
            vertex.x = vertex.x - middle.x;
            vertex.y = vertex.y - middle.y;
            vertex.z = vertex.z - middle.z;
        }

        var material = new THREE.MeshPhongMaterial({color: 0xff8000});

        var object = new THREE.Mesh(geometry, material);
        object.rotation.x = -Math.PI / 2;
        object.rotation.z = Math.PI / 2;
        scene.add(object);

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();
    }

    function writeOBJ(vertices, triangles) {
        var output = "";

        for (var coord of vertices) {
            output += "v " + coord[0] + " " + coord[1] + " " + coord[2] + "\n";
        }
        for (var triangle of triangles) {
            output += "f " + (triangle[0] + 1) + " " + (triangle[1] + 1) + " " + (triangle[2] + 1) + "\n";
        }

        return output;
    }

    function exportFile(text) {
        var element = document.createElement("a");
        element.setAttribute("href", "data:text/plain;charset=utf-8," + encodeURIComponent(text));
        element.setAttribute("download", filename);

        element.style.display = "none";
        document.body.appendChild(element);

        element.click();

        document.body.removeChild(element);
    }

});
