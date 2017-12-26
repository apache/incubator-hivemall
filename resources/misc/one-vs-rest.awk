#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

BEGIN{ FS="\t"; OFS="\t"; }
{
	possible_labels=$1;
	rowid=$2;
	label=$3;
	features=$4;

	label_count = split(possible_labels, label_array, ",");
 	
	for(i = 1; i <= label_count; i++) {
		if (label_array[i] == label)
			print rowid, label, 1, features;
		else
			print rowid, label_array[i], -1, features;
	}
}
END{}
